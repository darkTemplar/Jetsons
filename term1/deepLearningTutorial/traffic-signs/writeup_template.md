#**Traffic Sign Recognition** 

---

**Build a Traffic Sign Recognition Project**

The goals / steps of this project are the following:
* Load the data set (see below for links to the project data set)
* Explore, summarize and visualize the data set
* Design, train and test a model architecture
* Use the model to make predictions on new images
* Analyze the softmax probabilities of the new images
* Summarize the results with a written report


[//]: # (Image References)

[image1]: ./data/data_analysis/original_label_frequency_distribution.png "Original label frequency distribution"
[image2]: ./data/data_analysis/original_training_image.png "Original Image"
[image3]: ./data/data_analysis/preprocessed_image.png "Pre Processed Image"
[image4]: ./data/data_analysis/augmented_label_frequency_distribution.png "Augmented label frequency distribution"
[image5]: ./data/web_traffic_images/traffic-sign-keep-right.jpeg "Traffic Sign 1"
[image6]: ./data/web_traffic_images/traffic-sign-roadwork.jpeg "Traffic Sign 2"
[image7]: ./data/web_traffic_images/traffic-signs-bumpy_road.jpeg "Traffic Sign 3"
[image8]: ./data/web_traffic_images/traffic-sign-noentry.jpeg "Traffic Sign 4"
[image9]: ./data/web_traffic_images/traffic-sign-speed-limit.jpeg "Traffic Sign 5"

## Rubric Points
###Here I will consider the [rubric points](https://review.udacity.com/#!/rubrics/481/view) individually and describe how I addressed each point in my implementation.  

---
You're reading it! and here is a link to my [project code](https://github.com/udacity/CarND-Traffic-Sign-Classifier-Project/blob/master/Traffic_Sign_Classifier.ipynb)

###Data Set Summary & Exploration

####1. Provide a basic summary of the data set and identify where in your code the summary was done. In the code, the analysis should be done using python, numpy and/or pandas methods rather than hardcoding results manually.

The code for this step is contained in the second code cell of the IPython notebook.  

I used the numpy library to calculate summary statistics of the traffic
signs data set:

* The size of training set is 34799
* The size of test set is 12630
* The shape of a traffic sign image is (32, 32, 3)
* The number of unique classes/labels in the data set is 43

####2. Include an exploratory visualization of the dataset and identify where the code is in your code file.

The code for this step is contained in the third code cell of the IPython notebook.  

Here is an exploratory visualization of the data set. It is a bar chart showing the distribution of data by various class labels. This clearly shows that the training samples are unevenly distributed among the 43 classes which could mean the NN could be inadequately trained to identify those classes.

To reinforce the point above, We also notice that there are 21 labels which have less than 500 training examples under them (Code to generate that is also in the third cell). 

![Original label frequency distribution][image1]

###Design and Test a Model Architecture

####1. Describe how, and identify where in your code, you preprocessed the image data. What tecniques were chosen and why did you choose these techniques? Consider including images showing the output of each preprocessing technique. Pre-processing refers to techniques such as converting to grayscale, normalization, etc.

Data Preprocessing - The code for this step is contained in the fourth code cell of the IPython notebook. 

As a first step, I decided to convert the images to grayscale because color did not seem to play a huge role in determining what most of the signs meant and hence simplying the input space would possibly make it easier to train NN models. 

Secondly, we centered data around mean. Did not divide by std. deviation as image values are anyway within a given range so re-scaling probably not necessary.

Here is an example of a traffic sign image before and after preprocessing.

![Original Image][image2]

![Pre Processed Image][image3]


####2. Describe how, and identify where in your code, you set up training, validation and testing data. How much data was in each set? Explain what techniques were used to split the data into these sets. (OPTIONAL: As described in the "Stand Out Suggestions" part of the rubric, if you generated additional data for training, describe why you decided to generate additional data, how you generated the data, identify where in your code, and provide example images of the additional data)

I was already provided with different data sets for training, validating and testing. So there was no explicit need to split them up.  

My training set had 34799 number of images. My validation set and test set had 4410 and 12630 number of images.

The 4th cell in the notebook also code for augmenting the data set. I took 4 passes through the data set and decided via a simulated coin flip if I were to apply small amounts of translation, rotation and scaling to a given image. Used open cv3 to perform the above operations. We noted earlier that some labels were under represented i.e. had fewer than 500 training examples. For such labels we ignored the coin flip rule and just went ahead and added random peturbations (translation, rotation and scaling) so that we could have more training examples for such labels.


This ended up giving us a training set size of ~150k images. So now would be a good time to do another frequency distribution across labels of the dataset. We observe that now the dataset has no labels with less than 500 training examples. 

![Augmented Label Frequency Distribution][image4]


####3. Describe, and identify where in your code, what your final model architecture looks like including model type, layers, layer sizes, connectivity, etc.) Consider including a diagram and/or table describing the final model.

The code for my final model is located in the seventh cell of the ipython notebook. 

My final model consisted of the following layers:

| Layer         		|     Description	        					| 
|:---------------------:|:---------------------------------------------:| 
| Input         		| 32x32x1 Grayscale image   							| 
| Convolution 5x5     	| 1x1 stride, valid padding, outputs 28x28x38 	|
| RELU					|												|
| Max pooling	      	| 2x2 stride, same padding, outputs 14x14x38				|
| Convolution 5x5	    | 1x1 stride, valid padding, outputs 10x10x64  |
| RELU		|        									|
| Max pooling	      	| 2x2 stride, same padding, outputs 5x5x64				|
|	Fully Connected					|		inputs 1600, outputs 400										|
|	Fully Connected					|		inputs 400, outputs 200										|
|	Output					|		inputs 200, outputs 43									|
 


####4. Describe how, and identify where in your code, you trained your model. The discussion can include the type of optimizer, the batch size, number of epochs and any hyperparameters such as learning rate.

The code for training the model is located in the eigth cell of the ipython notebook. 

To train the model, I used an 20 epochs and a batch size of 128. The learning rate for backprop was 0.001. For the optimizer I used the AdamOptimizer sincelately it's the most widely used optimizer and has a built in bias correction mechanism.

For the actual training process, I followed the standard procedure for tensorflow based NN's i.e. build a computation graph describing the NN structure described above and then running the session with the inputs provided.

####5. Describe the approach taken for finding a solution. Include in the discussion the results on the training, validation and test sets and where in the code these were calculated. Your approach may have been an iterative process, in which case, outline the steps you took to get to the final solution and why you chose those steps. Perhaps your solution involved an already well known implementation or architecture. In this case, discuss why you think the architecture is suitable for the current problem.

The code for calculating the accuracy of the model is located in the ninth cell of the Ipython notebook.

My final model results were:
* training set accuracy of ~99%
* validation set accuracy of ~96-97%
* test set accuracy of 

I tried an iterative approach to building out the network:
* The first archhitecture that I tried was the Le Net-5 architecture (originally used for identifying MNIST dataset)
* The initial architecture however topped out on accuracy around ~88% and judging by the plot of training accuracy versus validation accuracy (x-axis being epochs) it seemed like we needed more parameters to capture features of the traffic signs which were probably more complicated than just single numeric digits.
* So I first started out by increasing the depth of the feature sets used in the convolutional layers (both 1 & 2). For this I played around with various numbers, drawing guidance from the baseline paper for solving the German Traffic sign by Yann Le Cunn and Pierre Sermanet. After various experiments, I finally settled on a depth of 38 for the first convolutional layer and 64 for the second convolutional layer. This seemed to get the accuracy up to about 93-94%. Experimented increasing the depth in both layers (doubled it) but that did not lead to any gains on the accuracy of the validation set.
* Next I decided to play around with the learning rate and epochs as I noticed that my validation accuracy would top out around epoch 15 and after that just hover nearby (indicating learning had stopped). I first tried decreasing the learning rate (by a factor of 0.5 each time), but that led to a big fall in validation accuracy so i ramped it back to 0.001. After that I played around with the number of epochs but increasing it to about 50 (from the original 20) did not seem to have any great effect on the accuracy.
* I next tried increasing the number of nodes in the fully connected layers, the original Lenet-5 architecture had 120 and 84 nodes respectively. I kept on increasing the number of nodes till I saw signs of overfitting happening (using the training error versus validation error graph referenced earlier). I finally fixed the nodes in the first fully connected layer to 400 and in the second connected layer to be 200. I tried testing dropout in both layers too when I had increased the nodes but that only seemed to drop validation accuracy.

Would have liked to try some other things like sending lower level output from the first two convolutional layers to the fully connected layer. However the speed of my experiments were hampered greatly by AWS failing to increase the service limit for a GPU instance and hence me having to run all my experiments on a CPU (MacBook Air) which was very time consuming.
 

###Test a Model on New Images

####1. Choose five German traffic signs found on the web and provide them in the report. For each image, discuss what quality or qualities might be difficult to classify.

Here are five German traffic signs that I found on the web:

![Keep Right][image5] ![Road Work][image6] ![Bumpy Road][image7] 
![No Entry][image8] ![Speed Limit 60][image9]

All of the images have some degree of diffculty involved in identifying them. Either they have some fuzziness or text overlaying the image or there is other stuff in the image (like the bumpy road sign also asks to reduce speed and the no entry sign is not facing the camera but has a different perspective)

####2. Discuss the model's predictions on these new traffic signs and compare the results to predicting on the test set. Identify where in your code predictions were made. At a minimum, discuss what the predictions were, the accuracy on these new predictions, and compare the accuracy to the accuracy on the test set (OPTIONAL: Discuss the results in more detail as described in the "Stand Out Suggestions" part of the rubric).

The code for making predictions on my final model is located in the tenth cell of the Ipython notebook.

Here are the results of the prediction:

| Image			        |     Prediction	        					| 
|:---------------------:|:---------------------------------------------:| 
| Keep Right      		| Stop sign   									| 
| Road Work     			| U-turn 										|
| Bumpy Road					| Yield											|
| 60 km/h	      		| Bumpy Road					 				|
| No Entry			| Slippery Road      							|


The model was able to correctly guess 4 of the 5 traffic signs, which gives an accuracy of 80%. This compares favorably to the accuracy on the test set of ...

####3. Describe how certain the model is when predicting on each of the five new images by looking at the softmax probabilities for each prediction and identify where in your code softmax probabilities were outputted. Provide the top 5 softmax probabilities for each image along with the sign type of each probability. (OPTIONAL: as described in the "Stand Out Suggestions" part of the rubric, visualizations can also be provided such as bar charts)

The code for making predictions on my final model is located in the 11th cell of the Ipython notebook.

For the first image, the model is relatively sure that this is a stop sign (probability of 0.6), and the image does contain a stop sign. The top five soft max probabilities were

| Probability         	|     Prediction	        					| 
|:---------------------:|:---------------------------------------------:| 
| .60         			| Stop sign   									| 
| .20     				| U-turn 										|
| .05					| Yield											|
| .04	      			| Bumpy Road					 				|
| .01				    | Slippery Road      							|


For the second image ... 
