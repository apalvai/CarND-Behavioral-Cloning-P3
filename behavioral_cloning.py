import csv
import numpy as np
import cv2

from keras.models import Sequential
from keras.layers import Flatten, Dense, Lambda, Cropping2D, Dropout
from keras.layers.convolutional import Convolution2D
from keras.layers.pooling import MaxPooling2D
from keras.layers.advanced_activations import LeakyReLU
from keras.callbacks import EarlyStopping

# fetch data
print('reading data...')
lines = []
with open ('./data/driving_log.csv') as csvfile:
    data = csv.reader(csvfile)
    for line in data:
        lines.append(line)

# remove the first row
lines = lines[1:]

images = []
measurements = []
for line in lines:
    
    # read center image
    center_image_filename = line[0].split('/')[-1]
    center_image_path = './data/IMG/' + center_image_filename
    image = cv2.imread(center_image_path)
    images.append(image)
    
    # read measurement
    measurement = float(line[3])
    measurements.append(measurement)
    
    # flipped center image and it's measurement
    flipped_image = np.fliplr(image)
    images.append(flipped_image)
    measurement_flipped = -measurement
    measurements.append(measurement_flipped)
    
    # correction
    correction = 0.1
    
    # read left image
    left_image_filename = line[1].split('/')[-1]
    left_image_path = './data/IMG/' + left_image_filename
    left_image = cv2.imread(left_image_path)
    images.append(left_image)
    
    # read left measurement
    left_measurement = measurement + correction
    measurements.append(left_measurement)
    
    # flipped left image and it's measurement
    flipped_left_image = np.fliplr(left_image)
    images.append(flipped_left_image)
    left_measurement_flipped = -left_measurement
    measurements.append(left_measurement_flipped)
    
    # read right image
    right_image_filename = line[2].split('/')[-1]
    right_image_path = './data/IMG/' + right_image_filename
    right_image = cv2.imread(right_image_path)
    images.append(right_image)
    
    # read right measurement
    right_measurement = measurement - correction
    measurements.append(right_measurement)
    
    # flipped right image and it's measurement
    flipped_right_image = np.fliplr(right_image)
    images.append(flipped_right_image)
    right_measurement_flipped = -right_measurement
    measurements.append(right_measurement_flipped)

X_train = np.array(images)
y_train = np.array(measurements)

print('X_train shape', X_train.shape)
print('y_train shape', y_train.shape)

input_shape = X_train[0].shape

# model
model = Sequential()
model.add(Cropping2D(cropping=((70, 25), (0, 0)), input_shape=input_shape))
model.add(Lambda(lambda x: (x / 255.0) - 0.5, input_shape=(90, 320, 3))) #Image normalization
model.add(Convolution2D(24, 5, 5, subsample=(2, 2)))
model.add(LeakyReLU(alpha=.001))
model.add(Convolution2D(36, 3, 3, subsample=(2, 2)))
model.add(LeakyReLU(alpha=.001))
model.add(Convolution2D(48, 3, 3, subsample=(2, 2)))
model.add(LeakyReLU(alpha=.001))
model.add(Convolution2D(64, 3, 3))
model.add(LeakyReLU(alpha=.001))
model.add(Convolution2D(64, 3, 3))
model.add(LeakyReLU(alpha=.001))
model.add(Flatten())
model.add(Dense(100))
model.add(LeakyReLU(alpha=.001))
model.add(Dense(50))
model.add(LeakyReLU(alpha=.001))
model.add(Dense(10))
model.add(LeakyReLU(alpha=.001))
model.add(Dense(1))

# training & validation
print('training...')
model.compile(loss='mse', optimizer='adam', metrics=['accuracy'])
#early_stopping = EarlyStopping(monitor='val_loss', patience=2)
model.fit(X_train, y_train, validation_split=0.2, shuffle=True, nb_epoch=3, verbose=1) # callbacks=[early_stopping]

# save model
print('saving model...')
model.save('model.h5')
