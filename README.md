![GitHub Repo stars](https://img.shields.io/github/stars/rppradhan08/face-mask-detection)
![GitHub forks](https://img.shields.io/github/forks/rppradhan08/face-mask-detection?color=green)
![contributors-shield](https://img.shields.io/github/contributors/rppradhan08/face-mask-detection)
[![LinkedIn][linkedin-shield]](https://in.linkedin.com/in/raj-praveen-pradhan-306625101)

[linkedin-shield]: https://img.shields.io/badge/-LinkedIn-black.svg?style=flat-square&logo=linkedin&colorB=555

<h1 align="center">
  <br>
  <a href="https://github.com/rppradhan08/face-mask-detection"><img src="https://raw.githubusercontent.com/rppradhan08/face-mask-detection/master/images/Facemask-Detection_2.jpg" alt="Face Mask Detector" width="400" style="border-radius: 50px"></a>
  <br>
  Real-time Face Mask Detector
  <br>
</h1>

<!-- TABLE OF CONTENTS -->

## Table of Contents

- [About the Project](#about-the-project)
  - [Approach](#approach)
  - [Steps Involved](#steps-involved)
- [Getting Started](#getting-started)
  - [Installations](#installations)
  - [Loading images and data preparation for the CNN classifier](#loading-images-and-data-preparation-for-the-cnn-classifier)
  - [Building and training CNN classifier](#building-and-training-cnn-classifier)
  - [Evaluating the CNN Classifier](#evaluating-the-cnn-classifier)
  - [Capturing video frames and detecting faces](#capturing-video-frames-and-detecting-faces)
  - [Performing perdictions and displaying results](#performing-perdictions-and-displaying-results)
- [Contacts](#contacts)

# About the Project

The World Health Organization (WHO) report suggests that respiratory droplets are one of the main routes of transmission of the COVID-19 virus. In a medical study, it was proven that surgical masks can prevent up to 95% of viral transmission caused due to respiratory droplets. This project uses Computer-Vision techniques like Face Detection and CNN classification to detect whether a person is wearing a mask in real-time.

## Approach

To detect whether the person is wearing a face mask in real-time, the app needs to keep track of individual faces per frame. OpenCV comes with many pre-trained DNN models for facial detection. Once faces are detected with high confidence, they are sent to a CNN classifier that detects face masks.

## Steps Involved

1. Loading images and preparing data for the CNN classifier
2. Building and training CNN classifier
3. Evaluating the CNN Classifier
4. Capturing video frames and detecting faces
5. Performing predictions and displaying results

# Getting Started

## Installations

Firstly, execute the below commands in the terminal for setting up the virtual environment and installing packages:

1. Create a virtual environment

```
python3 -m venv env
```

2. Activate newly created virtual environment `env`

```
env/Scripts/activate.bat
```

3. Execute the below command to install python packages used in this project

```
pip install requirement.txt
```

4. After installing all dependencies execute below command to run `main_app.py`.

```
python main_app.py
```

## Loading images and data preparation for the CNN classifier

To training our CNN classifier, images are present inside the `data.zip` file. Once unzipped, below will be the folder structure.

```
    data
     ├── with_mask  - this folder contains images of people with mask
     └── without_mask - this folder contains images of people without the mask
```

The below bar plot shows that labels are uniformly distributed. As there is a minimal class imbalance, 'accuracy' can be used as an evaluation metric for the classifier.

<h1 align="center">
  <br>
  <a href="https://github.com/rppradhan08/face-mask-detection"><img src="https://raw.githubusercontent.com/rppradhan08/face-mask-detection/master/images/class_distribution.png" alt="class_distribution" width="400" style="border-radius: 50px"></a>
</h1>

Once images are loaded, they are resized to a fixed size of 224x224 pixels.

<h1 align="center">
  <br>
  <a href="https://github.com/rppradhan08/face-mask-detection"><img src="https://raw.githubusercontent.com/rppradhan08/face-mask-detection/master/images/mask_nomask.png" alt="image_grid" width="400" style="border-radius: 50px"></a>
</h1>

The resized images are then normalized and passed through a data augmentation phase so that the model can generalize. Finally, 20% of the data is spared for testing the final model.

## Building and training CNN classifier

For building, the CNN classifier architecture transfer learning has been used to reduce the number of epochs to train the model. The [MobileNet](https://arxiv.org/abs/1704.04861) model was used as its base architecture, due to its high performance and smaller footprint. As the model will be capturing similar abstract features present in the `Imagenet` dataset, the trainable parameters of the Conv-Maxpool blocks are set to `False`. The top layers of MobileNet are chopped off and custom fully connected layers are added.
On, training the model for 20 epochs below are the loss and Accuracy curves for our train and test data.

After model completion, it is trained for 20 epochs to achieve an accuracy of `97.5%` on the train and test data. Below are the `accuracy` and `loss` of the trained model.

<h1 align="center">
  <br>
  <a href="https://github.com/rppradhan08/face-mask-detection"><img src="https://raw.githubusercontent.com/rppradhan08/face-mask-detection/master/images/Loss and accuracy curves.png" alt="metric_curves" width="400" style="border-radius: 50px"></a>
</h1>

## Evaluating the CNN Classifier

The trained model is evaluated on the test data. The below confusion matrix shows that the model fares pretty well on unseen data.

<h1 align="center">
  <br>
  <a href="https://github.com/rppradhan08/face-mask-detection"><img src="https://raw.githubusercontent.com/rppradhan08/face-mask-detection/master/images/confusion matrix.png" alt="confusion_metrics" width="400" style="border-radius: 50px"></a>
</h1>

## Capturing video frames and detecting faces

For capturing video frames, the `VideoCapture` module of OpenCV has been used. Then individual frames are passed through an OpenCV pre-trained DNN model for detecting faces present inside a frame. Faces that are above a certain threshold are saved and the rest are discarded. The model returns the pixel coordinates of detected faces. These coordinates are then used to extract faces and form the bounding box.

## Performing perdictions and displaying results

The faces captured by the face detector are resized and preprocessed so that they can be fed to the CNN model to classify whether the person is wearing a mask or not. Finally, the bounding and the resulting predictions are displayed on the captured frame.

<h1 align="center">
  <br>
  <a href="https://github.com/rppradhan08/face-mask-detection"><img src="https://raw.githubusercontent.com/rppradhan08/face-mask-detection/master/images/outcome.png" alt="outcome" width="400" style="border-radius: 50px"></a>
</h1>

# Contacts

Socials : [Linkedin](https://www.linkedin.com/in/raj-praveen-pradhan-306625101/)<br>
E-mail : [rppradhan310@gmail.com](rppradhan310@gmail.com)
