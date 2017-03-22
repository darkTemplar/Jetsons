import csv
import cv2
import numpy as np
from sklearn.utils import shuffle
from keras.models import Sequential
from keras.layers import Dense, Flatten, Lambda, Convolution2D, MaxPooling2D, Dropout
from keras.layers.convolutional import Cropping2D


with open('data/driving_log.csv') as csvfile:
    reader = csv.reader(csvfile)
    lines = [line for line in reader]

print("Extracting image and label (steering angle) data")
print("Number of images before augmentation ", len(lines))

from sklearn.model_selection import train_test_split
train_samples, validation_samples = train_test_split(lines, test_size=0.2)

BATCH_SIZE = 32
CAMERA_ANGLES = 3
KEEP_PROB = 0.6
steering_angles = []


def data_generator(samples, batch_size=32, training=True):
    soft_correction, hard_correction = 0.1, 0.2
    num_samples = len(samples)
    images, angles = np.zeros((batch_size*CAMERA_ANGLES, 160, 320, 3)), np.zeros((batch_size*CAMERA_ANGLES, 1))
    while 1: # Loop forever so the generator never terminates
        for i in range(0, batch_size, CAMERA_ANGLES):
            idx = np.random.randint(0, num_samples)
            sample = samples[idx]
            steering_angle = float(sample[3])
            steering_angles.append(steering_angle)
            # since there is a high skew towards straight driving in most tracks,
            # we probabilistically drop points for which steering angle is 0.0
            if training and steering_angle == 0.0 and np.random.uniform() > KEEP_PROB:
                continue
            # 0 - Center camera, 1 - Left camera and 2 - Right Camera
            for camera in range(CAMERA_ANGLES):
                img_source_path = sample[camera]
                img_filename = img_source_path.split('/')[-1]
                img_current_path = 'data/IMG/' + img_filename
                img = cv2.imread(img_current_path)
                if camera > 0:
                    # if right turn keep right camera image and steering angle adjustment
                    if steering_angle > 0.05:
                        if camera == 1:
                            correction = hard_correction if steering_angle > 0.2 else soft_correction
                            images[i+camera] = img
                            angles[i+camera] = steering_angle + correction
                            # since we won't be attaching left camera image, we flip the right one and use that instead
                            images[i+camera+1] = np.fliplr(img)
                            angles[i+camera+1] = -1. * (steering_angle + correction)

                    elif steering_angle < -0.05:
                        if camera == 2:
                            correction = hard_correction if steering_angle < -0.2 else soft_correction
                            images[i+camera] = img
                            angles[i+camera] = steering_angle - correction
                            # since we won't be attaching left camera image, we flip the right one and use that instead
                            images[i+camera-1] = np.fliplr(img)
                            angles[i+camera-1] = -1. * (steering_angle - correction)

                else:
                    images[i+camera] = img
                    angles[i+camera] = steering_angle
        yield images, angles



# compile and train the model using the generator function
train_generator = data_generator(train_samples, batch_size=BATCH_SIZE)
validation_generator = data_generator(validation_samples, batch_size=BATCH_SIZE, training=False)

#print("Number of images after augmentation ", len(images))

# Steering angle prediction CNN model
print("Defining Model in Keras")
model = Sequential()
# cropping layer
model.add(Cropping2D(cropping=((50, 20), (40, 40)), input_shape=(160, 320, 3)))
model.add(Lambda(lambda x: (x / 255.0) - 0.5))
model.add(Convolution2D(24, 5, 5, subsample=(2, 2)))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Convolution2D(36, 5, 5, subsample=(2, 2)))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Convolution2D(64, 3, 3, subsample=(1, 1)))
model.add(Flatten())
model.add(Dense(100))
model.add(Dense(10))
model.add(Dense(1))

print("Compile and run model")
model.compile(loss='mse', optimizer='adam')
model.fit_generator(train_generator, steps_per_epoch=len(train_samples)//BATCH_SIZE,
                    validation_data=validation_generator, validation_steps=len(validation_samples)//BATCH_SIZE, epochs=10)
model.save('model.h5')
