# Behaviorial Cloning Project

---
The goals / steps of this project are the following:
* Used the data provided by Udacity for training purposes
* Used convolutional neural network based on NVIDIA's architecture with 5 convolutional layers followed by 4 Fully connected layers
* Used MSE (Mean Squared Error) as the loss function and Adam optimizer to train the weigths
* Model was able to reduce the loss function ~0.026 after 7 epochs
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

The behavioral_cloning.py contains code for reading the data provided by Udacity, define, train and saving the model.
* reading the data collected from training
* splitting the data between training & validation samples (first 80% of data is used for training and the remaining 20% for validation)
* Convolutional neural network based off of NVIDIA's architecture with some minor modifications

###Model Architecture and Training Strategy

####1. Used Convolutional neural network based off of NVIDIA's architecture
I chose Convolutional Neural Network to clone the driving behavior as it can learn different features of the environment without needing too many parameters.

It consists of 5 convolutional layers followed by 4 fully connected layers with ELU activation and 2 layers of Dropout.
This model performs pretty well for the behavioral cloning task of driving around Track1.

Convolutional Layer: 3, 1, 1
Convolutional Layer: 24, 5, 5
ELU Activation: alpha = 0.1
Convolutional Layer: 36, 3, 3
ELU Activation: alpha = 0.1
Convolutional Layer: 48, 3, 3
ELU Activation: alpha = 0.1
Dropout: keep_prob = 0.5
Convolutional Layer: 64, 3, 3
ELU Activation: alpha = 0.1
Convolutional Layer: 64, 3, 3
ELU Activation: alpha = 0.1
Dropout: keep_prob = 0.5
Fully Connected Layer: 1164
ELU Activation: alpha = 0.1
Fully Connected Layer: 100
ELU Activation: alpha = 0.1
Fully Connected Layer: 50
ELU Activation: alpha = 0.1
Fully Connected Layer: 10
ELU Activation: alpha = 0.1
Output 

####2. Parameter tuning
I used Adam optimizer so as to avoid tuning the learning rate manually

####3. Training data
* Used 80-20 split of the available data for training and validation samples
* Trainging data has been shuffled for every epoch
* Used center, left and right cameras' images with equal probability. This also helps in training the model for recovery driving. For left and right cameras' images, I used an offset of +0.25 amd -0.25 respectively for steering angle measurements.  
* Employed Python's generator to augment the available training data by applying image processing techniques

#####1. Pre-processed images
** augment image brightness
** translate the image in x & y direction
** add rectangilar shadows in random areas of image 
** flip the processed image horizontally with a probability of 0.5 
** crop to remove unwanted areas (30 pixels from top and 20 pixels from bottom) 
** normalize and mean center it around 0
** resize the image to 64x64 to reduce the training times

#####2. Sampling data
** Looking at the distribution of steering angles in the training data, most of the data is near 0 which might result in model being trained to have bias towards 0. In order to avoid this scenario, I discarded some of the samples around 0 to have a better ditribution of steering angles.

