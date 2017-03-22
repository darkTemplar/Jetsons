import csv
import cv2
import numpy as np
import matplotlib.pyplot as plt


with open('data/driving_log.csv') as csvfile:
    reader = csv.reader(csvfile)
    lines = [line for line in reader]

steering_angles = []
BATCH_SIZE = 32
CAMERA_ANGLES = 3
KEEP_PROB = 0.6
# probability we will retain 0 steering angle images
keep_prob = 0.6
num_samples = len(lines)
images, angles = [], []

print("Extracting image and label (steering angle) data")
print("Number of images before augmentation ", num_samples)
for i, line in enumerate(lines):
    soft_correction, hard_correction = 0.1, 0.2
    steering_angle = float(line[3])
    steering_angles.append(steering_angle)
    # since there is a high skew towards straight driving in most tracks,
    # we probabilistically drop points for which steering angle is 0.0
    if steering_angle == 0.0 and np.random.uniform() > KEEP_PROB:
        continue
        # 0 - Center camera, 1 - Left camera and 2 - Right Camera
    for camera in range(CAMERA_ANGLES):
        img_source_path = line[camera]
        img_filename = img_source_path.split('/')[-1]
        img_current_path = 'data/IMG/' + img_filename
        img = cv2.imread(img_current_path)
        if camera > 0:
            # if right turn keep right camera image and steering angle adjustment
            if steering_angle > 0.05:
                if camera == 1:
                    correction = hard_correction if steering_angle > 0.2 else soft_correction
                    images.append(img)
                    angles.append(steering_angle + correction)
                    # since we won't be attaching left camera image, we flip the right one and use that instead
                    images.append(np.fliplr(img))
                    angles.append(-1. * (steering_angle + correction))

                elif steering_angle < -0.05:
                    if camera == 2:
                        correction = hard_correction if steering_angle < -0.2 else soft_correction
                        images.append(img)
                        angles.append(steering_angle - correction)
                        # since we won't be attaching left camera image, we flip the right one and use that instead
                        images.append(np.fliplr(img))
                        angles.append(-1. * (steering_angle - correction))

                else:
                    images.append(img)
                    angles.append(steering_angle)


# plot raw steering angles histogram
plt.hist(steering_angles, bins=30)
plt.title('Raw 3 cameras Steering angles histogram')
plt.show()
# plot sanitized steering angles histogram
plt.hist(angles, bins=30)
plt.title('Sanitized 3 cameras Steering angles histogram')
plt.show()


print("Number of images after sanitization ", len(images))

