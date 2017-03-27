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
FLIP_PROB = 0.5
# probability we will retain 0 steering angle images
keep_prob = 0.6
num_samples = len(lines)
images, angles = [], []
angles_dropped = []
zero_angles_dropped = []
print("Extracting image and label (steering angle) data")
print("Number of images before augmentation ", num_samples)
for i, line in enumerate(lines):
    soft_correction, hard_correction = 0.1, 0.2
    steering_angle = float(line[3])
    steering_angles.append(steering_angle)
    # since there is a high skew towards straight driving in most tracks,
    # we probabilistically drop points for which steering angle is 0.0
    if steering_angle == 0.0 and np.random.uniform() > KEEP_PROB:
        zero_angles_dropped.append(steering_angle)
        continue
        # 0 - Center camera, 1 - Left camera and 2 - Right Camera
    for camera in range(CAMERA_ANGLES):
        img_source_path = line[camera]
        img_filename = img_source_path.split('/')[-1]
        img_current_path = 'data/IMG/' + img_filename
        img = cv2.imread(img_current_path)
        if camera > 0:
            # if right turn, keep left camera image and steering angle adjustment
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
                angles_dropped.append(steering_angle)
        else:
            images.append(img)
            angles.append(steering_angle)
            if steering_angle != 0.0 and (steering_angle > 0.01 or steering_angle < -0.01):
                img = np.fliplr(img)
                steering_angle *= -1.
                images.append(img)
                angles.append(steering_angle)


# plot raw steering angles histogram
plt.hist(zero_angles_dropped, bins=40)
plt.title('Raw 3 cameras Steering angles histogram')
plt.show()
# plot sanitized steering angles histogram
plt.hist(angles, bins=40)
plt.title('Sanitized 3 cameras Steering angles histogram')
plt.show()

print("Number of images after sanitization ", len(images))

# generate examples of processing & distortions on images
# 1. Cropping
"""img_file = 'examples/center_camera.jpg'
img = cv2.imread(img_file)
cv2.imshow("original", img)
crop_img = img[50:140, 40:280]
cv2.imshow("cropped", crop_img)
cv2.imwrite('examples/crop_center_camera.jpg', crop_img)
cv2.waitKey(0)

#2.Flipping
img_file = 'examples/crop_center_camera.jpg'
img = cv2.imread(img_file)
cv2.imshow("original", img)
flipped_img = np.fliplr(img)
cv2.imshow("flipped", flipped_img)
cv2.imwrite('examples/flipped_center_camera.jpg', flipped_img)
cv2.waitKey(0)

# Gaussian Blurring
img_file = 'examples/crop_center_camera.jpg'
img = cv2.imread(img_file)
cv2.imshow("original", img)
blurred_img = cv2.GaussianBlur(img, (5, 5), 0)
cv2.imshow("blurred", blurred_img)
cv2.imwrite('examples/blurred_center_camera.jpg', blurred_img)
cv2.waitKey(0)

# adding jitter
h,w,c = img.shape # (768, 1024, 3)

noise = np.random.randint(0,50,(h, w)) # design jitter/noise here
zitter = np.zeros_like(img)
zitter[:,:,1] = noise

noise_added = cv2.add(img, zitter)
combined = np.vstack((img[:h/2,:,:], noise_added[h/2:,:,:]))

cv2.imshow(combined)"""
