import csv
import numpy as np
import cv2
import random
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.utils import shuffle

from keras.preprocessing.image import ImageDataGenerator

from keras.models import Sequential
from keras.layers import Flatten, Dense, Lambda, Cropping2D, Dropout
from keras.layers.convolutional import Convolution2D
from keras.layers.pooling import MaxPooling2D
from keras.layers.advanced_activations import ELU
from keras.callbacks import EarlyStopping

# fetch data
print('reading data...')
samples = []
with open ('./data/driving_log.csv') as csvfile:
    data = csv.reader(csvfile)
    for line in data:
        samples.append(line)

# remove the first row
samples = samples[1:]

# split training and validation data
train_samples, validation_samples = train_test_split(samples, test_size=0.2)

# read image
def read_image(path):
    image = cv2.imread(path)
    image = cv2.cvtColor(image,cv2.COLOR_BGR2RGB)
    return image

# flip image and steering angle
def flip_image_and_measurement(image, angle):
    flipped_image = np.fliplr(image)
    flipped_angle = -angle
    return flipped_image, flipped_angle

# augment image brightness
def augment_brightness(img):
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV) #convert it to hsv
    hsv = np.array(hsv, dtype = np.float64)
    h, s, v = cv2.split(hsv)
    v = v*(0.5 + random.uniform(0, 1))
    v[v>255] = 255
    final_hsv = cv2.merge((h, s, v))
    final_hsv = np.array(final_hsv, dtype = np.uint8)
    img = cv2.cvtColor(final_hsv, cv2.COLOR_HSV2BGR)
    return img

# add gaussian noise
def add_gaussian_noise(image):
    row,col,ch= image.shape
    mean = 0
    var = 0.1
    sigma = var**0.5
    gauss = np.random.normal(mean,sigma,(row,col,ch))
    gauss = gauss.reshape(row,col,ch)
    noisy = image + gauss
    return noisy

# translate image and adjust sterring angle
def translate(image, angle, trans_x_range=100, trans_y_range=40, angle_range=0.2):
    trans_x = trans_x_range * (random.uniform(0, 1) - 0.5)
    trans_y = trans_y_range * (random.uniform(0, 1) - 0.5)
    updated_angle = angle + (trans_x/trans_x_range) * 2 * angle_range

    #translation matrix
    trans_matrix = np.float32([[1, 0, trans_x], [0, 1, trans_y]])

    # Apply transformation to translate the image
    trans_image = cv2.warpAffine(image, trans_matrix, (image.shape[1], image.shape[0]))
    
    return trans_image, updated_angle

# add random shadow
def add_random_shadow(image):
    img_height, img_width, channels = image.shape
    x_offset = random.randint(0, img_width)
    y_offset = random.randint(0, img_height)
    
    width = random.randint(int(img_width/2), img_width)
    if(x_offset + width > img_width):
        x_offset = img_width - x_offset
    
    height = random.randint(int(img_height/2), img_height)
    if(y_offset + height > img_height):
        y_offset = img_height - y_offset

    image[y_offset:y_offset + height, x_offset:x_offset + width, 2] = image[y_offset:y_offset + height, x_offset:x_offset + width, 2] * 0.25
    return image

# pre-processing of image
def process_image_and_measurement(image, angle):
    
    image = augment_brightness(image)
    
    if random.uniform(0, 1) > 0.3:
        image = add_gaussian_noise(image)
    
    if random.uniform(0, 1) > 0.4:
        image = add_random_shadow(image)

    if random.uniform(0, 1) > 0.2:
        image, angle = translate(image, angle)

    if random.uniform(0, 1) > 0.5:
        image, angle = flip_image_and_measurement(image, angle)
    
    return image, angle

#plot steering angle measurements distribution
def plot_steering_angles(y_train):
    unique_angles = list(set(y_train))
    unique_angles = unique_angles.sort()
    plt.hist(y_train, unique_angles)
    print('saving plot...')
    plt.savefig('steering_angle_distribution.png', bbox_inches='tight')
    # plt.show()


# data generator
def generator(samples, batch_size=32):
    num_samples = len(samples)
    
    while 1:
        shuffle(samples)
        
        for offset in range(0, num_samples, batch_size):
            batch_samples = samples[offset:offset+batch_size]
            
            images = []
            angles = []
            
            for batch_sample in batch_samples:
                clr = random.randint(0, 2)
                
                # correction
                correction = 0.25
                image_column_index = 0
                
                if clr == 0:
                    # cneter camera image
                    image_column_index = 0
                    steering_adjustment = 0.0
                elif clr == 1:
                    # left camera image
                    col_index = 1
                    steering_adjustment = correction
                elif clr == 2:
                    # right camera image
                    image_column_index = 2
                    steering_adjustment = -correction
                
                # read image & steering angle
                image_path = './data/IMG/' + batch_sample[image_column_index].split('/')[-1]
                image = read_image(image_path)
                angle = float(batch_sample[3]) + steering_adjustment
                
                # discard some of the data centered around 0 steering angle
                keep_prob = 0
                while keep_prob == 0:
                    # process image and acordingly steering angle
                    processed_image, updated_angle = process_image_and_measurement(image, angle)
                    threshold = random.uniform(0, 1)
                    if abs(updated_angle) < 0.10:
                        val = random.uniform(0, 1)
                        if val > threshold:
                            keep_prob = 1
                    else:
                        keep_prob = 1
                
                # add image and steering angle to respective arrays
                images.append(processed_image)
                angles.append(updated_angle)
                
            X_train = np.array(images)
            y_train = np.array(angles)
            yield shuffle(X_train, y_train)

# generators for training and validation
train_generator = generator(train_samples, batch_size=32)
validation_generator = generator(validation_samples, batch_size=32)

# plot steering angle measurements
#steering_measurements = []
#for i in range(50):
#    steering_measurements.extend((next(train_generator))[1])
#plot_steering_angles(steering_measurements)


# params
elu_alpha = 0.1
keep_prob = 0.5

# model
model = Sequential()
model.add(Cropping2D(cropping=((60, 20), (0, 0)), input_shape=(160, 320, 3))) # Image cropping
model.add(Lambda(lambda x: (x / 255.0) - 0.5, input_shape=(80, 320, 3))) #Image normalization
model.add(Convolution2D(3, 1, 1, border_mode='same'))
model.add(Convolution2D(24, 5, 5, subsample=(2, 2)))
model.add(ELU(elu_alpha))
model.add(Dropout(keep_prob))
model.add(Convolution2D(36, 3, 3, subsample=(2, 2)))
model.add(ELU(elu_alpha))
model.add(Dropout(keep_prob))
model.add(Convolution2D(48, 3, 3, subsample=(2, 2)))
model.add(ELU(elu_alpha))
model.add(Dropout(keep_prob))
model.add(Convolution2D(64, 3, 3))
model.add(ELU(elu_alpha))
model.add(Dropout(keep_prob))
model.add(Convolution2D(64, 3, 3))
model.add(ELU(elu_alpha))
model.add(Dropout(keep_prob))
model.add(Flatten())
model.add(Dense(1164))
model.add(ELU(elu_alpha))
model.add(Dropout(keep_prob))
model.add(Dense(100))
model.add(ELU(elu_alpha))
model.add(Dropout(keep_prob))
model.add(Dense(50))
model.add(ELU(elu_alpha))
model.add(Dropout(keep_prob))
model.add(Dense(10))
model.add(ELU(elu_alpha))
model.add(Dropout(keep_prob))
model.add(Dense(1))

# training & validation
print('training...')
model.compile(loss='mse', optimizer='adam')
model.fit_generator(train_generator,
                    samples_per_epoch=len(train_samples)*6,
                    validation_data=validation_generator,
                    nb_val_samples=len(validation_samples),
                    nb_epoch=3,
                    verbose=1)

#model.fit_generator(train_generator.flow(X_train, y_train, batch_size=32),
#                    samples_per_epoch=len(X_train)*2,
#                    validation_data=validation_generator.flow(X_valid, y_valid, batch_size=32),
#                    nb_val_samples=len(X_valid),
#                    nb_epoch=3)

# save model
print('saving model...')
model.save('model.h5')
