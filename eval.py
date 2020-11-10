import torch
import torchvision
import torch.nn as nn
import torch.utils.data as data
import torchvision.transforms as trans
from torch.utils.data import DataLoader, random_split
from PIL import Image
import time
import pandas as pd
import numpy as np
import glob
from utils import cosine_anneal_schedule, load_model, jigsaw_generator
import csv
from Model import PMG, BasicConv


device = "cuda:0"


# create 2 dict to mapname to number and map number to name
label_csv = pd.read_csv('training_labels.csv')
label = label_csv.values[:, 1]
label_set = set(label)
label_list = list(label_set)
label2number = {}
number2label = {}
for i in range(len(label_set)):
    label2number[label_list[i]] = i
    number2label[i] = label_list[i]


def load_img(filepath):
    img = Image.open(filepath)
    # convert to RGB mode
    img = img.convert('RGB')
    return img


class TestDataset(data.Dataset):
    def __init__(self, input_transform):
        super(TestDataset, self).__init__()

        self.file_list = glob.glob("testing_data/testing_data/*.jpg")
        self.input_transform = input_transform

    def __getitem__(self, index):
        file_name = self.file_list[index]
        image = load_img(file_name)
        if self.input_transform:
            image = self.input_transform(image)
        return image, file_name

    def __len__(self):
        return len(self.file_list)


test_trans = trans.Compose([
    trans.Resize((448, 448), Image.BILINEAR),
    trans.ToTensor(),
    trans.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])
test_dataset = TestDataset(input_transform=test_trans)
test_loader = DataLoader(dataset=test_dataset,
                         batch_size=1,
                         shuffle=False)


# evaluation with trained model, and output prediction to csv file
net = torch.load("PMG.pth").to(device)
net.eval()
y_pred = []
file_names = []
correct = 0
with torch.no_grad():
    for i, (inputs, file_name) in enumerate(test_loader):
        inputs = inputs.to(device)
        file_names.append(file_name)

        output_1, output_2, output_3, output_concat = net(inputs)
        outputs_com = output_1 + output_2 + output_3 + output_concat

        _, predicted_com = torch.max(outputs_com.data, 1)
        y_pred.append(predicted_com.item())


# write prediction into csv file
with open('answer.csv', 'w', newline='') as file:
    csv_writer = csv.writer(file)
    csv_writer.writerow(["id", "label"])
    for idx, i in enumerate(y_pred):
        csv_writer.writerow([file_names[idx][0][26:32], number2label[i]])
