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
My model consists of 5 convolutional layers followed by 4 fully connected layers with ELU activation and 2 layers of Dropout. This model performs pretty well for the behavioral cloning task of driving around Track1.

I chose Convolutional Neural Network(CNN) to clone the driving behavior as it can learn different features of the environment without needing too many parameters. CNNs are particular useful when dealing with images as they allow us to optimize the architecture which in-turn will result in lesser number of parameters. Nvidia's architecture seemed liked a balance once between using too many layers (over-fitting) versus very few layers (under-fitting). I have employed initial convolution layer for model to chose color space[2]. I used ELU activation function (with alpha=0.001), as it seemed to allow model to learn faster with better accuracy than the other ones like RELU, Leaky RELU. To avoid over-fitting and make it more generalized, i've added 2 layers of Dropout with 0.5 probability. With this network i've achieved a training loss of ~0.267 after 7 epochs.

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

I save the model locally and re-train by tweaking some parameters to improve the performance.

####2. Parameter tuning
* I used Adam optimizer so as to avoid tuning the learning rate manually.
* I have tried various values of alpha for ELU and found 0.001 to have better accuracy (resulting smoother driving).
* I have used different combinations of Dropout layers and noticed that adding more of them increased the training loss. Hence 2 Droput layers seemed like compromise between making a model generalized and achieve desired accuracy.
* I have added 2 additional convolutional layers of size (128, 3, 3) but did not see any considerable improvement in reducing the training/validation loss, so i used the same number of layers as described in NVIDIA's paper.  

####3. Training data
* Used 80-20 split of the available data for training and validation samples
* Trainging data has been shuffled for every epoch
* Used center, left and right cameras' images with equal probability. This also helps in training the model for recovery driving. For left and right cameras' images, I used an offset of +0.25 amd -0.25 respectively for steering angle measurements.  
* Employed Python's generator to augment the available training data by applying image processing techniques and used around ~50k samples.

#####1. Image processing techniques
** Augment image brightness: This will enable model to be more generalized and perform well even during varying light conditions.
** Translate the image in x & y direction: This will help in recovery driving
** Add shadows (reactangular) in random areas of image 
** Flip the processed image horizontally with a probability of 0.5 
** Crop to remove unwanted areas (30 pixels from top and 20 pixels from bottom) 
** Normalize and mean center it around 0
** Resize the image to 64x64 to reduce the training times

#####2. Sampling data
** Looking at the distribution of steering angles in the training data, most of the data is near 0 which might result in model being trained to have bias towards 0. In order to avoid this scenario, I discarded some of the samples around 0 to have a better ditribution of steering angles.

### Discussion
This project was challenging and a great learning experience. Picking the right # of training epochs was some what counter-intutive as I would expect the model to learn well as you increase the # of epochs. Although my model was able to drive around Track1, it fails to complete Track2. I've' tried with different combinations but my model's training loss did not go beyond ~0.0267. I've used different image augmentation techniques to make the model perform better in various lighting conditions and recover well if it deviates from the center of road. However, I suspect there is a penalty in the form of increased training loss. I've made use of Python's Generator concept to create more samples than available, using different image processing techniques and also able to train the model in a memory efficient way.

### References
1. http://cs231n.github.io/
2. https://chatbotslife.com/using-augmentation-to-mimic-human-driving-496b569760a9
3. https://medium.com/@mohankarthik/cloning-a-car-to-mimic-human-driving-5c2f7e8d8aff
4. https://keras.io/
5. https://github.com/fchollet/keras/issues/688
6. http://image-net.org/challenges/posters/JKU_EN_RGB_Schwarz_poster.pdf
7. http://www.picalike.com/blog/2015/11/28/relu-was-yesterday-tomorrow-comes-elu/
8. https://www.quora.com/How-does-ELU-activation-function-help-convergence-and-whats-its-advantages-over-ReLU-or-sigmoid-or-tanh-function
9. https://arxiv.org/abs/1511.07289
10. https://www.quora.com/How-does-the-dropout-method-work-in-deep-learning
