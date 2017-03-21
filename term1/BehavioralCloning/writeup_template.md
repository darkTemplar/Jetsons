---

**Behavioral Cloning Project**

The goals / steps of this project are the following:
* Use the simulator to collect data of good driving behavior
* Build, a convolution neural network in Keras that predicts steering angles from images
* Train and validate the model with a training and validation set
* Test that the model successfully drives around track one without leaving the road
* Summarize the results with a written report


[//]: # (Image References)

[image1]: ./examples/placeholder.png "Model Visualization"
[image2]: ./examples/placeholder.png "Grayscaling"
[image3]: ./examples/placeholder_small.png "Recovery Image"
[image4]: ./examples/placeholder_small.png "Recovery Image"
[image5]: ./examples/placeholder_small.png "Recovery Image"
[image6]: ./examples/placeholder_small.png "Normal Image"
[image7]: ./examples/placeholder_small.png "Flipped Image"

## Rubric Points
###Here I will consider the [rubric points](https://review.udacity.com/#!/rubrics/432/view) individually and describe how I addressed each point in my implementation.  

---
###Files Submitted & Code Quality

####1. Submission includes all required files and can be used to run the simulator in autonomous mode

My project includes the following files:
* model.py containing the script to create and train the model
* drive.py for driving the car in autonomous mode
* model.h5 containing a trained convolution neural network 
* writeup_report.md summarizing the results

####2. Submission includes functional code
Using the Udacity provided simulator and my drive.py file, the car can be driven autonomously around the track by executing 
```sh
python drive.py model.h5
```

####3. Submission code is usable and readable

The model.py file contains the code for training and saving the convolution neural network. The file shows the pipeline I used for training and validating the model, and it contains comments to explain how the code works.

####4. Training data accumulation and collection

Training data was chosen to keep the vehicle driving on the road. I drove for about 10-11 laps on track 1 and mostly tried to keep the car on center lane. 

###5.Model Architecture and Training Strategy

I started by using the Nvidia self driving CNN (as shown below) used to predict steering angles as my base network. 

This was a regression model and the error measure being used was MSE.

In order to gauge how well the model was working, I split my image and steering angle data into a training and validation set. I found that my first model had a low mean squared error on the training set (0.0047) but a relatively higher mean squared error on the validation set (0.0062). This implied that the model was overfitting. 

To combat that, I modified my model in the following ways:

i) Increase the number of training epochs from 5 to 20 in steps of 5 (so tried 10, 15, 20). 15 seemed to work best given the training data.
ii) Increase batch size from default of 32 in keras to 64, 128
iii) Reduced the number of nodes in the first fully connected layer from 1164 to 500
iv) Removed 2 Convolutional layers
v) Add max pooling to the existing Convolutional layers and added dropout to the fully connected layer

All of the above steps led to a decrease in validation loss dropping it to around 0.0028 (Corresponding training loss was 0.0017)

Based on the above the final architecture of the model was as follows.

My final model consisted of the following layers:

| Layer         		|     Description	        					| 
|:---------------------:|:---------------------------------------------:| 
| Input         		| 160x320x3 image   							| 
| Cropping         		| Crop non-helpful parts of the image   							| 
| Normalization     	| Normalize intensity and mean center images 	|
| Convolution					|												|
| Max pooling	      	| 2x2 stride, same padding, outputs 14x14x38				|
| Convolution	    | 1x1 stride, valid padding, outputs 10x10x64  |
| Max pooling	      	| 2x2 stride, same padding, outputs 5x5x64				|
| Convolution	    | 1x1 stride, valid padding, outputs 10x10x64  |
| Max pooling	      	| 2x2 stride, same padding, outputs 5x5x64				|
| Flatten	      	| 2x2 stride, same padding, outputs 5x5x64				|
|	Fully Connected					|		inputs 500, outputs 200										|
|	Dropout					|		dropout prob. = 50%										|
|	Fully Connected					|		inputs 200, outputs 100										|
|	Fully Connected					|		inputs 200, outputs 100										|
|	Output					|		inputs 1, outputs 1 (steering angle)								|



The final step was to run the simulator to see how well the car was driving around track one. Despite the low validation and training accuracy, there were a couple of spots where the vehicle fell off the track (esp. at sharp turn after the bridge). Possible causes in my mind were 
i) My driving during training was not spot on esp. with regards to recovery driving (I had some difficulty in doing that). To see if this was true, I decided to use the sample data provided by Udacity on my existing model for training and validation and finally to have it drive autonomously.

ii) The other solution since the car was failing at a couple of sharp turns was to try and take into account the input of the left and right cameras as well. This would require me to use fit_generator offered by keras instead of fit since I was running into memory issues if I included images from all 3 cameras.

Here are my results from the above approaches.

i) The udacity data set had 8036 training images (after flipping that increased to 16072). 

####3. Creation of the Training Set & Training Process

To capture good driving behavior, I recorded two laps on track one using center lane driving. Here is an example image of center lane driving:

![alt text][image2]

I then recorded the vehicle recovering from the left side and right sides of the road back to center so that the vehicle would learn to .... These images show what a recovery looks like starting from ... :

![alt text][image3]
![alt text][image4]
![alt text][image5]

Then I repeated this process on track two in order to get more data points.

To augment the data sat, I also flipped images and angles thinking that this would ... For example, here is an image that has then been flipped:

![alt text][image6]
![alt text][image7]

Etc ....

After the collection process, I had X number of data points. I then preprocessed this data by ...


I finally randomly shuffled the data set and put Y% of the data into a validation set. 

I used this training data for training the model. The validation set helped determine if the model was over or under fitting. The ideal number of epochs was Z as evidenced by ... I used an adam optimizer so that manually training the learning rate wasn't necessary.
