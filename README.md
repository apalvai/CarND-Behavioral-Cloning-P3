# Behaviorial Cloning Project

---
The goals / steps of this project are the following:
* Used the data provided by Udacity for training purposes
* Used convolutional neural network based on NVIDIA's architecture with 5 convolutional layers followed by 4 Fully connected layers
* Used MSE (Mean Squared Error) as the loss function and Adam optimizer to train the weigths
* Model was able to reduce the loss function ~0.015 after 3 epochs
* Used the above trained model to drive the car around track1 on simuator and model performed reasonably well.
* However, the model was not able to drive track2 as it is more challenging.
* I believe the model was not generalized enough to drive itself on track2 based on training on track1. 

---
###Files Submitted
####1. Submission includes all required fields and can be used to run the simuator in autonomous mode
My project includes following files:
* behvaioral_cloning.py for defining and training the model
* drive.py for driving the car in autonomous mode in udacity's simulator
* model.h5 trained model
* run-001.mp4 & run-002.mp4 video files demonstrating the performance of the model in autonomous driving mode
* write_up.md report about the details of project


####2. Submission includes functional code Using the Udacity provided simulator and my drive.py file, the car can be driven autonomously around the track by executing

python drive.py model.h5

####3. Submission code is usable and readable

The behavioral_cloning.py contains the code for reading the data provided by udacity, define, train and save the model.
* reading the data collected from training
* splitting the data between training & validation samples
* Convolutional neural network based off of NVIDIA's architecture

###Model Architecture and Training Strategy

####1. Used Convolutional neural network based off of NVIDIA's architecture
It consists of 5 convolutional layers followed by 4 fully connected layers
Convolutional Layer: 24, 5, 5
Leaky RELU Activation: alpha = 0.001
Convolutional Layer: 36, 3, 3
Leaky RELU Activation: alpha = 0.001
Convolutional Layer: 48, 3, 3
Leaky RELU Activation: alpha = 0.001
Convolutional Layer: 64, 3, 3
Leaky RELU Activation: alpha = 0.001
Convolutional Layer: 64, 3, 3
Leaky RELU Activation: alpha = 0.001
Fully Connected Layer: 1164
Leaky RELU Activation: alpha = 0.001
Fully Connected Layer: 100
Leaky RELU Activation: alpha = 0.001
Fully Connected Layer: 50
Leaky RELU Activation: alpha = 0.001
Fully Connected Layer: 10
Leaky RELU Activation: alpha = 0.001
Output

####2. Parameter tuning
I used Adam optimizer so as to avoid tuning the learning manually

####3. Training data
Used 80-20 split of the available data for training and validation samples
Trainging data has been suffled for every epoch
Used left and right cameras' images as well for recovery driving
Pre-processed images
* flip horizontally
* crop to remove unwanted areas
* normalize and mean center it around 0
