**Behavioral Cloning** 

---

**Behavioral Cloning Project**

The goals / steps of this project are the following:
* Use the simulator to collect data of good driving behavior
* Build, a convolution neural network in Keras that predicts steering angles from images
* Train and validate the model with a training and validation set
* Test that the model successfully drives around track one without leaving the road
* Summarize the results with a written report


[//]: # (Image References)

[image1]: ./examples/nvidia_cnn.png "Nvidia CNN"
[image2]: ./examples/raw_steering_angles_hist.png "Central Camera Raw Steering Angles Histogram"
[image3]: ./examples/raw_3_cameras_steering_angles.png "3 Cameras Raw Steering Angles Histogram"
[image4]: ./examples/center_camera.jpg "Center camera"
[image5]: ./examples/crop_center_camera.jpg "Cropped Image"
[image6]: ./examples/flipped_center_camera.jpg "Flipped Image"
[image7]: ./examples/blurred_center_camera.jpg "Blurred Image"
[image8]: ./examples/left_camera.jpg "Left camera Image"
[image9]: ./examples/right_camera.jpg "Right camera Image"
[image10]: ./examples/balanced_3_camera_steering_angles.png "Balanced 3 Cameras Steering Angles Histogram"


---
Files Submitted & Code Quality

1. Submission includes all required files and can be used to run the simulator in autonomous mode

My project includes the following files:
* model.py containing the script to create and train the model
* drive.py for driving the car in autonomous mode
* model.h5 containing a trained convolution neural network 
* behavioral_cloning_report.md summarizing the results

2. Submission includes functional code
Using the Udacity provided simulator and my drive.py file, the car can be driven autonomously around the track by executing 
```sh
python drive.py model.h5
```

3. Submission code is usable and readable

The model.py file contains the code for training and saving the convolution neural network. The file shows the pipeline I used for training and validating the model, and it contains comments to explain how the code works.

4. Training data collection

Training data was chosen to keep the vehicle driving on the road. I drove for about 10-11 laps on track 1 and mostly tried to keep the car on center lane. Initially we only considered the center camera, this led to about 11k images. More details on training data augmentation and modification are in Section 6 (Autonomous Driving and Training Process). Below is an example of a center camera image and the cropped version which is the first layer of our keras model.

![Center Camera][image4]

![Cropped][image5]

5.Model Architecture and Training Strategy

I started by using the Nvidia self driving CNN (as shown below) used to predict steering angles as my base network. This was implemented using Keras (TensorFlow backend). I added ReLu activation to the convolutional layers to introduce non-linearity into the model. 

![Nvidia CNN][image1]

This was a regression model and the error measure being used was MSE.

In order to gauge how well the model was working, I split my image and steering angle data into a training and validation set (80/20) with shuffling. I found that my first model had a low mean squared error on the training set (0.0087) but a relatively higher mean squared error on the validation set (0.0127). This implied that the model was overfitting. 

To combat that, I modified my model in the following ways:

i) Increase the number of training epochs from 5 to 20 in steps of 5 (so tried 10, 15, 20). Used ModelCheckpoint in Keras to save the best model in each case (criteria being minimizing validation loss). Most of the time the best loss was achieved by epoch 10, so final model uses 10 epochs.

ii) Increase batch size from default of 32 in keras to 64, 128, 256. Once again ModelCheckpoint indicated no significant validation loss reduction was achieved by increasing batch size from 32.

iii) Remove one of the fully connected layers (the one with 50 noedes, as can be seen in the NVIDIA CNN architecture above).

iv) Removed 2 Convolutional layers

v) Add max pooling to the existing Convolutional layers and added dropout to the fully connected layer. The dropout on fully connected layers did not increase validation accuracy so ended up removing it. Also ended up removing max pooling on the last convolutional layer.

All of the above steps led to a decrease in validation loss dropping it to around 0.0058 (Corresponding training loss was 0.0048)

Based on the above the final architecture of the model was as follows.

My final model consisted of the following layers:

| Layer         		|     Description	        					| 
|:---------------------:|:---------------------------------------------:| 
| Input         		  | 160x320x3 image   							| 
| Cropping         		| Crop outputs 90x240x3   							| 
| Normalization     	| Normalize intensity and mean center images 	|
| Convolution 5x5					|	2x2 stride, valid padding, outputs 43x118x24											|
| RELU         		    |   							|
| Max pooling	      	| 2x2 stride, valid padding, outputs 21x59x24				|
| Convolution	5x5        | 2x2 stride, valid padding, outputs 9x28x36  |
| RELU         		    |   							|
| Max pooling	      	| 2x2 stride, valid padding, outputs 4x14x36				|
| Convolution	3x3        | 1x1 stride, valid padding, outputs 1x12x64  |
| RELU         		    |   							|
| Flatten	      	    | 768 nodes				|
|	Fully Connected			|	outputs 100										|
|	Fully Connected			|	outputs 10										|
|	Output					    |	outputs 1 (steering angle)								|


6. Autonomous Driving and Training Process

The final step was to run the simulator to see how well the car was driving around track one. Despite the low validation and training accuracy, there were a couple of spots where the vehicle fell off the track and could not recover (esp. at sharp turn after the bridge). 

Probable cause in my mind was the fact that my driving during training was not spot on esp. with regards to recovery driving (I had some difficulty in doing that). To help understand this better, I decided to visualize the initial data set (see fig.below) (which I should have done earlier but I guess better late than never). 

![Central Camera Raw Steering Angles][image2]

This turned out to be quite an eye opener. As the histogram plot of the raw steering angles will reveal, vast majority of the track was probably straight line driving and hence steering angles were near 0.0. This would explain very well the fact that the car was unable to perform sharp turns. To alleviate this problem, I introduced a dropout like keep_probability for 0 steering angles (I experimented with different values and settled on 0.6). So I would only keep about 60% of the near 0 steering angle data points. This was only done during training. While this made the car perform slightly better but it was still driving off sharp turns. To alleviate this problem, I did 2 things

i) I added 4-5 more laps to the data set where I tried driving in recovery mode. This led to driving_log.csv having about 17k lines. In addition I added augmented data points by flippiong and blurring images for which steering angles were non-zero. Below are examples of images which were flipped and blurred (using the same crop central camera image displayed in section 4)

![Flipped][image6]

![Burred][image7]

This was done to increase data points where car was turning (See fig2.). This led to the data set looking slightly more balanaced. Somewhat counter intuitively, doing this actually worsened my loss on the training and validation data sets and the car still could not complete laps around track 1. I made various changes to the network (adding more layers, experimenting with max pooling and dropouts etc.) but could not get around this problem. This led me to believe that maybe using only the central camera was insufficient for the way i was approaching the problem.

ii) As a result of the above,  I started considering left and right camera angles also. See examples of raw left and right camera images below. 

![Left Camera][image8]

![Right Camera][image9]


As I experimented with the new camera angles, There was still a noticeable skew in the data due to straight line driving and also the left sided nature of track 1. This is shown below.

![3 Cameras Raw Steering Angles][image3]

To help overcome that, I refined my approach to only include right camera angle for left turns and vice-versa with the corresponding steering correction. This was done to make car perform better at sharp turns. Also when i included a camera angle and it's steering correction I used flipping to create a mirror of that image to balance out left handedness of the track. As a further optimization, I added a larger correction (0.2) for sharper turns and softer value (0.1) for smaller turns. To further augment training data esp. for non-zero steering angles, even for the central camera images I flipped images and added them to training set. So the above approach combined with the random reduction of 0 steering angle images using keep_probability (explained above) led to a much more balanced dataset (with balanced right and left steering angles and also reduced number of zero steering angles). This data set is visualized below and we can notice this has a far better distribution of steering angles compared to figure above.

![Balanced 3 Camera Steering Angles][image10]

This approach paid dividends and I training and validation loss dropped to around 0.0025-0.003. Finally, the car was able to navigate around track1 quite well. It still went off the center a couple of times but recovered well. I did this at various speeds (modifying controller speed value) in drive.py. Due to larger number of images and hence insuuficient memory, I used generators and the fit_generator function from Keras.

Finally, I trained and validated my model with my own dataset as well as the dataset provided by Udacity. The model was able to keep the car on track in the simulator in both cases.

7. Future Improvements
Make model generalize better.As part of that, we can do the following things:
i) Add more augmented data as after down sampling straight line driving data, the number of data points can become low. We could also just drive more laps.

ii) Test and run model on track2 and complete a lap there (right now it does not perform well on that). Probably need to drive a few test laps on track2 as well and train on that data to help model generalize better and make sure it's not just memorizing track 1.

iii) Also add a few training laps on different configurations of the simulator (simple graphics etc.) plus augment training data by changing brightness, adding jitter etc to account for running this model on different hardware settings. 


