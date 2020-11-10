import torch
import torchvision
import torch.nn as nn
import torch.optim as optim
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

# hyper-parameters
batch_size = 16
epochs = 50
device = "cuda:0"

# create 2 dict to mapname to number and map number to name
csv = pd.read_csv('training_labels.csv')
label = csv.values[:, 1]
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


def getTrainData():
    csv = pd.read_csv('training_labels.csv')
    img_id, label = csv.values[:, 0], csv.values[:, 1]
    return img_id, label


class CarDataset(data.Dataset):
    def __init__(self, input_transform):
        super(CarDataset, self).__init__()

        self.image_id, self.labels = getTrainData()
        self.input_transform = input_transform

    def __getitem__(self, index):
        file_name = "training_data/training_data/{:06d}.jpg".format(
                        self.image_id[index])
        image = load_img(file_name)
        if self.input_transform:
            image = self.input_transform(image)

        label_name = self.labels[index]
        label = label2number[label_name]
        return image, label

    def __len__(self):
        return len(self.image_id)


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


train_trans = trans.Compose([
    trans.Resize((448, 448), Image.BILINEAR),
    trans.RandomHorizontalFlip(),
    trans.ColorJitter(brightness=0.2, contrast=0.2),
    trans.ToTensor(),
    trans.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])
test_trans = trans.Compose([
    trans.Resize((448, 448), Image.BILINEAR),
    trans.ToTensor(),
    trans.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])

# split validation dataset from training dataset
# (80% for training, 20% for validation)

train_dataset = CarDataset(input_transform=train_trans)
# train_size = int(0.8 * len(labeled_dataset))
# valid_size = len(labeled_dataset) - train_size
# train_dataset, valid_dataset = random_split(labeled_dataset,
#                                     [train_size, valid_size])
# valid_dataset.input_transform = test_trans

test_dataset = TestDataset(input_transform=test_trans)

train_loader = DataLoader(dataset=train_dataset,
                          batch_size=batch_size,
                          shuffle=True)
# valid_loader = DataLoader(dataset=valid_dataset,
#                           batch_size=1,
#                           shuffle=False)
test_loader = DataLoader(dataset=test_dataset,
                         batch_size=1,
                         shuffle=False)

net = load_model(model_name='resnet50_pmg',
                 pretrain=True,
                 require_grad=True)
net = net.to(device)
criterion = nn.CrossEntropyLoss()
optimizer = optim.SGD([
        {'params': net.classifier_concat.parameters(), 'lr': 0.002},
        {'params': net.conv_block1.parameters(), 'lr': 0.002},
        {'params': net.classifier1.parameters(), 'lr': 0.002},
        {'params': net.conv_block2.parameters(), 'lr': 0.002},
        {'params': net.classifier2.parameters(), 'lr': 0.002},
        {'params': net.conv_block3.parameters(), 'lr': 0.002},
        {'params': net.classifier3.parameters(), 'lr': 0.002},
        {'params': net.features.parameters(), 'lr': 0.0002}],
        momentum=0.9, weight_decay=5e-4)
lr = [0.002, 0.002, 0.002, 0.002, 0.002, 0.002, 0.002, 0.0002]

for epoch in range(epochs):
    # trainging
    print("{:d} epoch:".format(epoch))
    net.train()
    train_loss = 0
    train_loss1 = 0
    train_loss2 = 0
    train_loss3 = 0
    train_loss4 = 0
    correct = 0
    total = 0
    idx = 0
    start = time.time()
    for batch_idx, (inputs, labels) in enumerate(train_loader):
        idx = batch_idx
        if inputs.shape[0] < batch_size:
            continue

        inputs, labels = inputs.to(device), labels.to(device)

        # update learning rate
        for nlr in range(len(optimizer.param_groups)):
            optimizer.param_groups[nlr]['lr'] = cosine_anneal_schedule(epoch, epochs, lr[nlr])

        # Step 1
        optimizer.zero_grad()
        inputs1 = jigsaw_generator(inputs, 8)
        output_1, _, _, _ = net(inputs1)
        loss1 = criterion(output_1, labels) * 1
        loss1.backward()
        optimizer.step()

        # Step 2
        optimizer.zero_grad()
        inputs2 = jigsaw_generator(inputs, 4)
        _, output_2, _, _ = net(inputs2)
        loss2 = criterion(output_2, labels) * 1
        loss2.backward()
        optimizer.step()

        # Step 3
        optimizer.zero_grad()
        inputs3 = jigsaw_generator(inputs, 2)
        _, _, output_3, _ = net(inputs3)
        loss3 = criterion(output_3, labels) * 1
        loss3.backward()
        optimizer.step()

        # Step 4
        optimizer.zero_grad()
        _, _, _, output_concat = net(inputs)
        concat_loss = criterion(output_concat, labels) * 2
        concat_loss.backward()
        optimizer.step()

        #  training log
        _, predicted = torch.max(output_concat.data, 1)
        total += labels.size(0)
        correct += predicted.eq(labels.data).cpu().sum()

        train_loss += (loss1.item() + loss2.item() + loss3.item() + concat_loss.item())
        train_loss1 += loss1.item()
        train_loss2 += loss2.item()
        train_loss3 += loss3.item()
        train_loss4 += concat_loss.item()

        if batch_idx % 50 == 0:
            print(
                'Step: %d | Loss1: %.3f | Loss2: %.3f | Loss3: %.3f | Loss_concat: %.3f | Loss: %.3f | Acc: %.3f%% (%d/%d)' % (
                batch_idx, train_loss1 / (batch_idx + 1), train_loss2 / (batch_idx + 1),
                train_loss3 / (batch_idx + 1), train_loss4 / (batch_idx + 1), train_loss / (batch_idx + 1),
                100. * float(correct) / total, correct, total))

    train_acc = 100. * float(correct) / total
    train_loss = train_loss / (idx + 1)
    end = time.time()
    print("Loss1: {:.3f}, Loss2: {:.3f}, Loss3: {:.3f}, Loss_concat: {:.3f}, train_acc: {:.3f}, time: {:.2f}"
        .format(train_loss1 / (idx + 1), train_loss2 / (idx + 1),
                train_loss3 / (idx + 1), train_loss4 / (idx + 1), train_acc, end-start))

    # validation
#     net.eval()
#     valid_loss = 0
#     correct = 0
#     correct_com = 0
#     total = 0
#     idx = 0
#     with torch.no_grad():
#         for batch_idx, (inputs, labels) in enumerate(valid_loader):
#             idx = batch_idx
#             inputs, labels = inputs.to(device), labels.to(device)
#             output_1, output_2, output_3, output_concat= net(inputs)
#             outputs_com = output_1 + output_2 + output_3 + output_concat

#             loss = criterion(output_concat, labels)

#             valid_loss += loss.item()
#             _, predicted = torch.max(output_concat.data, 1)
#             _, predicted_com = torch.max(outputs_com.data, 1)
#             total += labels.size(0)
#             correct += predicted.eq(labels.data).cpu().sum()
#             correct_com += predicted_com.eq(labels.data).cpu().sum()

#         valid_acc = 100. * float(correct) / total
#         valid_acc_en = 100. * float(correct_com) / total
#         valid_loss = valid_loss / (idx + 1)
#     print("valid loss: {:.3f}, valid acc: {:.3f}, valid acc combined: {:.3f}"
#          .format(valid_loss,valid_acc,valid_acc_en))

# save trained model
torch.save(net, "PMG.pth")


# evaluation with trained model, and output prediction to csv file
net = torch.load("PMG.pth").to(device)
net.eval()
y_pred = []
file_names = []
correct = 0
for i, (inputs, file_name) in enumerate(test_loader):
    inputs = inputs.to(device)
    file_names.append(file_name)

    output_1, output_2, output_3, output_concat = net(inputs)
    outputs_com = output_1 + output_2 + output_3 + output_concat

    _, predicted_com = torch.max(outputs_com.data, 1)
    y_pred.append(predicted_com.item())


# write prediction into csv file
with open('answer.csv', 'w', newline='') as file:
    writer = csv.writer(file)
    writer.writerow(["id", "label"])
    for idx, i in enumerate(y_pred):
        writer.writerow([file_names[idx][0][26:32], number2label[i]])
