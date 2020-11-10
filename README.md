# NCTU Selected Topics in Visual Recognition using Deep Learning, Homework 1
Code for [HW1](https://www.kaggle.com/t/14e99b9514d74996b6b04df4fed0ed19): fine-grained classification.


## Hardware
The following specs were used to create the submited solution.
- Ubuntu 16.04 LTS
- Intel(R) Core(TM) i9-9900K CPU @ 3.60GHz
- NVIDIA GeForce 2080Ti

## Reproducing Submission
To reproduct my submission without retrainig, do the following steps:
1. [Installation](#Installation)
2. [Dataset Preparation](#Dataset-Preparation)
3. [Prepapre Dataset](#Prepare-Dataset)
4. [Train models](#Train-models)
5. [Pretrained models](#Pretrained-models)
6. [Reference](#Reference)

## Installation
All requirements should be detailed in requirements.txt. Using Anaconda is strongly recommended.
```
conda create -n hw1 python=3.7
source activate hw1
pip install -r requirements.txt
```

## Dataset Preparation
Dataset download link is in Data section of [HW1](https://www.kaggle.com/t/14e99b9514d74996b6b04df4fed0ed19)

## Prepare Dataset
After downloading, the data directory is structured as:
```
${ROOT}
  +- testing_data
  |  +- testing_data
  |  |  +- 000004.jpg
  |  |  +- 000005.jpg
  |  |  +- ...
  +- training_data
  |  +- training_data
  |  |  +- 000001.jpg
  |  |  +- 000002.jpg
  |  |  +- ...
  +- training_labels

```

### Train models
To train models, run following commands.
```
$ python run.py 
```
It will train the model and output `answer.csv` which is the prediction of testing data.


## Pretrained models
You can download pretrained model that used for my submission from [link](https://drive.google.com/file/d/1ZXEZNwtoyDXSouYicKncoLLV1tizBmhO/view?usp=sharing).
And put it in the directory :
```
${ROOT}
  +- PMG.pth
  +- eval.py
```

To evaluation the pretrained model and get the prediction of testing data:
```
$ python eval.py 
```


## Reference
[Fine-Grained Visual Classiﬁcation via Progressive Multi-Granularity Training of Jigsaw Patches (ECCV2020)](https://github.com/PRIS-CV/PMG-Progressive-Multi-Granularity-Training)