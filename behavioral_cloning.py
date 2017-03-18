import csv
import numpy as np
import cv2

from keras.models import Sequential
from keras.layers import Flatten, Dense, Lambda

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
    # read image
    image_path = line[0]
    filename = image_path.split('/')[-1]
    current_path = './data/IMG/' + filename
    image = cv2.imread(current_path)
    images.append(image)

    # read measurement
    measurement = float(line[3])
    measurements.append(measurement)

X_train = np.array(images)
y_train = np.array(measurements)

print('X_train shape', X_train.shape)
print('y_train shape', y_train.shape)

input_shape = X_train[0].shape

# model
model = Sequential()
model.add(Lambda(lambda x: (x / 255.0) - 0.5, input_shape=input_shape))
model.add(Flatten())
model.add(Dense(1))

# training & validation
model.compile(loss='mse', optimizer='adam')
model.fit(X_train, y_train, validation_split=0.2, shuffle=True)

# save model
model.save('model.h5')
