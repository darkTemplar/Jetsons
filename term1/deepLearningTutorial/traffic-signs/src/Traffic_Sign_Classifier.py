
# coding: utf-8

# # Self-Driving Car Engineer Nanodegree
# 
# ## Deep Learning
# 
# ## Project: Build a Traffic Sign Recognition Classifier
# 
# In this notebook, a template is provided for you to implement your functionality in stages, which is required to successfully complete this project. If additional code is required that cannot be included in the notebook, be sure that the Python code is successfully imported and included in your submission if necessary. 
# 
# > **Note**: Once you have completed all of the code implementations, you need to finalize your work by exporting the iPython Notebook as an HTML document. Before exporting the notebook to html, all of the code cells need to have been run so that reviewers can see the final implementation and output. You can then export the notebook by using the menu above and navigating to  \n",
#     "**File -> Download as -> HTML (.html)**. Include the finished document along with this notebook as your submission. 
# 
# In addition to implementing code, there is a writeup to complete. The writeup should be completed in a separate file, which can be either a markdown file or a pdf document. There is a [write up template](https://github.com/udacity/CarND-Traffic-Sign-Classifier-Project/blob/master/writeup_template.md) that can be used to guide the writing process. Completing the code template and writeup template will cover all of the [rubric points](https://review.udacity.com/#!/rubrics/481/view) for this project.
# 
# The [rubric](https://review.udacity.com/#!/rubrics/481/view) contains "Stand Out Suggestions" for enhancing the project beyond the minimum requirements. The stand out suggestions are optional. If you decide to pursue the "stand out suggestions", you can include the code in this Ipython notebook and also discuss the results in the writeup file.
# 
# 
# >**Note:** Code and Markdown cells can be executed using the **Shift + Enter** keyboard shortcut. In addition, Markdown cells can be edited by typically double-clicking the cell to enter edit mode.

# ---
# ## Step 0: Load The Data

# In[13]:

# Load pickled data
import pickle

# TODO: Fill this in based on where you saved the training and testing data

training_file = 'train.p'
validation_file= 'valid.p'
testing_file = 'test.p'

with open(training_file, mode='rb') as f:
    train = pickle.load(f)
with open(validation_file, mode='rb') as f:
    valid = pickle.load(f)
with open(testing_file, mode='rb') as f:
    test = pickle.load(f)
    
X_train, y_train = train['features'], train['labels']
original_X_train, original_y_train = X_train[:5000], y_train[:5000]
X_valid, y_valid = valid['features'], valid['labels']
X_test, y_test = test['features'], test['labels']


# ---
# 
# ## Step 1: Dataset Summary & Exploration
# 
# The pickled data is a dictionary with 4 key/value pairs:
# 
# - `'features'` is a 4D array containing raw pixel data of the traffic sign images, (num examples, width, height, channels).
# - `'labels'` is a 1D array containing the label/class id of the traffic sign. The file `signnames.csv` contains id -> name mappings for each id.
# - `'sizes'` is a list containing tuples, (width, height) representing the the original width and height the image.
# - `'coords'` is a list containing tuples, (x1, y1, x2, y2) representing coordinates of a bounding box around the sign in the image. **THESE COORDINATES ASSUME THE ORIGINAL IMAGE. THE PICKLED DATA CONTAINS RESIZED VERSIONS (32 by 32) OF THESE IMAGES**
# 
# Complete the basic data summary below. Use python, numpy and/or pandas methods to calculate the data summary rather than hard coding the results. For example, the [pandas shape method](http://pandas.pydata.org/pandas-docs/stable/generated/pandas.DataFrame.shape.html) might be useful for calculating some of the summary results. 

# ### Provide a Basic Summary of the Data Set Using Python, Numpy and/or Pandas

# In[14]:

### Replace each question mark with the appropriate value. 
### Use python, pandas or numpy methods rather than hard coding the results

n_train = len(X_train)

n_test = len(X_test)

image_shape = X_train[0].shape

n_classes = len(set(y_train))

print("Number of training examples =", n_train)
print("Number of testing examples =", n_test)
print("Image data shape =", image_shape)
print("Number of classes =", n_classes)


# ### Include an exploratory visualization of the dataset

# Visualize the German Traffic Signs Dataset using the pickled file(s). This is open ended, suggestions include: plotting traffic sign images, plotting the count of each sign, etc.
# 
# The [Matplotlib](http://matplotlib.org/) [examples](http://matplotlib.org/examples/index.html) and [gallery](http://matplotlib.org/gallery.html) pages are a great resource for doing visualizations in Python.
# 
# **NOTE:** It's recommended you start with something simple first. If you wish to do more, come back to it after you've completed the rest of the sections.

# In[15]:

### Data exploration visualization code goes here.
### Feel free to use as many code cells as needed.
import matplotlib.pyplot as plt
import numpy as np
import cv2
# Visualizations will be shown in the notebook.
get_ipython().magic('matplotlib inline')

# plot frequencies of various classes (43 in total) in the training dataset
from collections import Counter
train_label_frequencies = dict(Counter(y_train))
plt.bar(list(train_label_frequencies.keys()), list(train_label_frequencies.values()), 1., color='r')
# print number of labels which have less than 500 examples
print("Number of label classes with less than 500 examples in training data ",sum([1 for k,v in train_label_frequencies.items() if v < 500]))


# ----
# 
# ## Step 2: Design and Test a Model Architecture
# 
# Design and implement a deep learning model that learns to recognize traffic signs. Train and test your model on the [German Traffic Sign Dataset](http://benchmark.ini.rub.de/?section=gtsrb&subsection=dataset).
# 
# There are various aspects to consider when thinking about this problem:
# 
# - Neural network architecture
# - Play around preprocessing techniques (normalization, rgb to grayscale, etc)
# - Number of examples per label (some have more than others).
# - Generate fake data.
# 
# Here is an example of a [published baseline model on this problem](http://yann.lecun.com/exdb/publis/pdf/sermanet-ijcnn-11.pdf). It's not required to be familiar with the approach used in the paper but, it's good practice to try to read papers like these.
# 
# **NOTE:** The LeNet-5 implementation shown in the [classroom](https://classroom.udacity.com/nanodegrees/nd013/parts/fbf77062-5703-404e-b60c-95b78b2f3f9e/modules/6df7ae49-c61c-4bb2-a23e-6527e69209ec/lessons/601ae704-1035-4287-8b11-e2c2716217ad/concepts/d4aca031-508f-4e0b-b493-e7b706120f81) at the end of the CNN lesson is a solid starting point. You'll have to change the number of classes and possibly the preprocessing, but aside from that it's plug and play!

# ### Pre-process the Data Set (normalization, grayscale, etc.)

# Use the code cell (or multiple code cells, if necessary) to implement the first step of your project.

# In[17]:

### Preprocess the data here. Preprocessing steps could include normalization, converting to grayscale, etc.
### Feel free to use as many code cells as needed.

import cv2

def augmentImageData(images, labels, passes=4):
    global train_label_frequencies
    num_examples = len(labels)
    new_images, new_labels = [], []
    translate, scale, rotate = False, False, False
    for i in range(passes):
        for j in range(num_examples):
            image, label = images[j], labels[j]
            rows,cols = image.shape[:2]
            # if none of the transformations apply AND label is not under represented class, move on to next image
            # if label is part of under represented i.e. less than 500 examples, we force augmentation for that image
            under_represented = train_label_frequencies.get(label, 0) < 500
            if under_represented:
                translate, scale, rotate = True, True, True
            else:
                # we do a coin flip to determine which transformations to apply
                translate, scale, rotate = np.random.uniform() > 0.5, np.random.uniform() > 0.5, np.random.uniform() > 0.5
            
            # if none of the transformations apply, move on to next image
            if not translate and not scale and not rotate:
                continue
            if translate:
                shift_x, shift_y = np.random.randint(-3,3), np.random.randint(-3,3)
                # create transformation matrix
                M = np.float32([[1,0,shift_x],[0,1,shift_y]])
                image = cv2.warpAffine(image,M,(cols,rows))
                
            if scale:
                scale_x, scale_y = np.random.uniform(1., 1.3), np.random.uniform(1., 1.3)
                image = cv2.resize(image, None, fx=scale_x, fy=scale_y, interpolation = cv2.INTER_CUBIC)
                # TODO: crop image to convert it back to 32 x 32
                image_height, image_width = image.shape
                padding_height, padding_width = image_height - 32, image_width - 32
                if padding_height % 2 == 0:
                    image = image[padding_height//2:image_height - padding_height//2, padding_width//2:image_width - padding_width//2] if padding_width % 2 == 0 else image[padding_height//2:image_height - padding_height//2, padding_width//2:image_width - (padding_width+1)//2]
                else:
                    image = image[padding_height//2:image_height - (padding_height+1)//2, padding_width//2:image_width - padding_width//2] if padding_width % 2 == 0 else image[padding_height//2:image_height - (padding_height+1)//2, padding_width//2:image_width - (padding_width+1)//2]
                
                
            if rotate:
                rotation_degrees = np.random.randint(-15,15)
                M = cv2.getRotationMatrix2D((cols/2,rows/2),rotation_degrees,1)
                image = cv2.warpAffine(image,M,(cols,rows))

            new_images.append(image[:,:,np.newaxis])
            new_labels.append(label)
    return np.concatenate([images, np.array(new_images)]), np.concatenate([labels, np.array(new_labels)])

def preProcessImages(images):   
    # Pre processing Step 1 - Turn images into grayscale
    num_images = len(images)
    gray_images = np.zeros((num_images, 32, 32, 1))
    for i in range(num_images):
        image = images[i]
        image = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
        image = image[:,:,np.newaxis]
        gray_images[i] = image
        
    # Pre processing Step 2 - Center and normalize the images
    gray_images -= np.mean(gray_images)
    return gray_images

# preprocess the training images
X_train = preProcessImages(X_train)
original_X_train = preProcessImages(original_X_train)
# preprocess the validation images
X_valid = preProcessImages(X_valid)

# visualize a randomly chosen training image
index = np.random.randint(0, len(X_train))
image = X_train[index].squeeze()
plt.imshow(image)

# augment training dataset
print("num of training examples BEFORE augmentation ", len(X_train))
X_train, y_train = augmentImageData(X_train, y_train)
print("num of training examples AFTER training ", len(X_train))

# re-visualize the labels to see if we increased examples in each class
# plot frequencies of various classes (43 in total) in the training dataset
train_label_frequencies = dict(Counter(y_train))
plt.bar(list(train_label_frequencies.keys()), list(train_label_frequencies.values()), 1., color='r')
# print number of labels which have less than 500 examples
print("Number of label classes with less than 500 examples in augmented training data ",sum([1 for k,v in train_label_frequencies.items() if v < 500]))


# ### Model Architecture

# In[8]:

### Define your architecture here.
### Feel free to use as many code cells as needed.
import tensorflow as tf

EPOCHS = 15
BATCH_SIZE = 128



from tensorflow.contrib.layers import flatten

def AbhasNet(x):
    global n_classes
    # Arguments used for tf.truncated_normal, randomly defines variables for the weights and biases for each layer
    mu = 0
    sigma = 0.1
    
    # Layer 1: Convolutional. Input = 32x32x1. Output = 28x28x38.
    w1 = tf.Variable(tf.truncated_normal([5,5,1,38], mean=mu, stddev=sigma))
    b1 = tf.Variable(tf.zeros(38))
    layer1 = tf.nn.conv2d(x, w1, strides=[1, 1, 1, 1], padding="VALID")
    layer1 = tf.nn.bias_add(layer1, b1)
    # Activation.
    layer1 = tf.nn.relu(layer1)
    # Pooling. Input = 28x28x38. Output = 14x14x38.
    layer1 = tf.nn.max_pool(layer1, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding="SAME")
    
    

    # Layer 2: Convolutional. Output = 10x10x64.
    w2 = tf.Variable(tf.truncated_normal([5,5,38,64], mean=mu, stddev=sigma))
    b2 = tf.Variable(tf.zeros(64))
    layer2 = tf.nn.conv2d(layer1, w2, [1, 1, 1, 1], padding="VALID")
    layer2 = tf.nn.bias_add(layer2, b2)
    # Activation.
    layer2 = tf.nn.relu(layer2)
    # Pooling. Input = 10x10x80. Output = 5x5x64.
    layer2 = tf.nn.max_pool(layer2, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding="SAME")
    # Flatten. Input = 5x5x64. Output = 1600.
    layer2 = flatten(layer2)
    
    # Layer 3: Fully Connected. Input = 1600. Output = 200.
    w3 = tf.Variable(tf.truncated_normal([1600, 200], mean=mu, stddev=sigma))
    b3 = tf.Variable(tf.zeros(200))
    layer3 = tf.add(tf.matmul(layer2, w3), b3)
    # Activation.
    layer3 = tf.nn.relu(layer3)
    
    # Layer 4: Fully Connected. Input = 200. Output = 100.
    w4 = tf.Variable(tf.truncated_normal([200, 100], mean=mu, stddev=sigma))
    b4 = tf.Variable(tf.zeros(100))
    layer4 = tf.add(tf.matmul(layer3, w4), b4)
    # Activation.
    layer4 = tf.nn.relu(layer4)
    
    # Layer 5: Fully Connected. Input = 100. Output = 43.
    w5 = tf.Variable(tf.truncated_normal([100, n_classes], mean=mu, stddev=sigma))
    b5 = tf.Variable(tf.zeros(n_classes))
    logits = tf.add(tf.matmul(layer4, w5), b5)
    print("Logits defined for ", n_classes)
    return logits


# ### Train, Validate and Test the Model

# A validation set can be used to assess how well the model is performing. A low accuracy on the training and validation
# sets imply underfitting. A high accuracy on the training set but low accuracy on the validation set implies overfitting.

# In[10]:

### Train your model here.
### Calculate and report the accuracy on the training and validation set.
### Once a final model architecture is selected, 
### the accuracy on the test set should be calculated and reported as well.
### Feel free to use as many code cells as needed.
 
# `x` is a placeholder for a batch of input images.
# `y` is a placeholder for a batch of output labels.
# `keep_prob` is a placeholder for dropout prob. for the fully connected layers.


x = tf.placeholder(tf.float32, (None, 32, 32, 1))
y = tf.placeholder(tf.int32, (None))
keep_prob3 = tf.placeholder(tf.float32)
#keep_prob4 = tf.placeholder(tf.float32)
one_hot_y = tf.one_hot(y, n_classes)


# ## Training Pipeline

rate = 0.001

logits = AbhasNet(x)
cross_entropy = tf.nn.softmax_cross_entropy_with_logits(logits=logits, labels=one_hot_y)
loss_operation = tf.reduce_mean(cross_entropy)
optimizer = tf.train.AdamOptimizer(learning_rate = rate)
training_operation = optimizer.minimize(loss_operation)


# ## Model Evaluation
# Evaluate how well the loss and accuracy of the model for a given dataset.

correct_prediction = tf.equal(tf.argmax(logits, 1), tf.argmax(one_hot_y, 1))
accuracy_operation = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
saver = tf.train.Saver()


def evaluate(X_data, y_data):
    num_examples = len(X_data)
    total_accuracy = 0
    sess = tf.get_default_session()
    for offset in range(0, num_examples, BATCH_SIZE):
        batch_x, batch_y = X_data[offset:offset+BATCH_SIZE], y_data[offset:offset+BATCH_SIZE]
        accuracy = sess.run(accuracy_operation, feed_dict={x: batch_x, y: batch_y})
        total_accuracy += (accuracy * len(batch_x))
    return total_accuracy / num_examples


# ## Train the Model
# Run the training data through the training pipeline to train the model.
# 
# Before each epoch, shuffle the training set.
# 
# After each epoch, measure the loss and accuracy of the validation set.
# 
# Save the model after training.
from sklearn.utils import shuffle
with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())
    num_examples = len(X_train)
    
    print("Training...")
    print()
    for i in range(EPOCHS):
        X_train, y_train = shuffle(X_train, y_train)
        for offset in range(0, num_examples, BATCH_SIZE):
            end = offset + BATCH_SIZE
            batch_x, batch_y = X_train[offset:end], y_train[offset:end]
            sess.run(training_operation, feed_dict={x: batch_x, y: batch_y})
        
        #training_accuracy = evaluate(original_X_train, original_y_train)
        print("EPOCH {} ...".format(i+1))
        #print("Training Accuracy = {:.3f}".format(training_accuracy))
        validation_accuracy = evaluate(X_valid, y_valid)
        print("Validation Accuracy = {:.3f}".format(validation_accuracy))
        print()
        
    saver.save(sess, './abhasnet')
    print("Model saved")

    
with tf.Session() as sess:
    # preprocess test set
    X_test = preProcessImages(X_test)
    saver.restore(sess, tf.train.latest_checkpoint('.'))
    test_accuracy = evaluate(X_test, y_test)
    print("Test Accuracy = {:.3f}".format(test_accuracy))


# ---
# 
# ## Step 3: Test a Model on New Images
# 
# To give yourself more insight into how your model is working, download at least five pictures of German traffic signs from the web and use your model to predict the traffic sign type.
# 
# You may find `signnames.csv` useful as it contains mappings from the class id (integer) to the actual sign name.

# ### Load and Output the Images

# In[42]:

### Load the images and plot them here.
### Feel free to use as many code cells as needed.
import glob, os
# we add the data type to the web traffic images declaration 
# so that it will play nice with cv2.cvtColor which is present in the preprocessing function
web_traffic_images = np.empty((5, 32, 32, 3), np.uint8)
idx = 0
for file in glob.glob("*.jpeg"):
    print("Reading file ", file)
    plt.imshow(image)
    plt.show()
    image = cv2.imread(file, 1)
    image = cv2.resize(image, (32, 32), 0, 0, interpolation = cv2.INTER_CUBIC)
    web_traffic_images[idx] = image
    idx += 1


# ### Predict the Sign Type for Each Image

# In[40]:

### Run the predictions here and use the model to output the prediction for each image.
### Make sure to pre-process the images with the same pre-processing pipeline used earlier.
### Feel free to use as many code cells as needed.
# ## Evaluate the Model
# Once you are completely satisfied with your model, evaluate the performance of the model on the test set.
# 
# Be sure to only do this once!
# 
# If you were to measure the performance of your trained model on the test set, then improve your model, and then measure the performance of your model on the test set again, that would invalidate your test results. You wouldn't get a true measure of how well your model would perform against real data.
# 
# You do not need to modify this section.

# run preprocessing on the 5 web traffic images
web_traffic_images = preProcessImages(web_traffic_images)

# add labels for the web traffic images in the following order:
# (keep right, no entry, road work, 60 speed limit, bumpy road)
y_web = np.array([38,17,25,3,24], np.uint8)
with tf.Session() as sess:
    saver.restore(sess, tf.train.latest_checkpoint('.'))
    web_test_accuracy = evaluate(web_traffic_images, y_web)
    print("Test Accuracy = {:.3f}".format(web_test_accuracy))
    softmax = tf.nn.softmax(logits=logits)
    top_k = sess.run(softmax, feed_dict={x: web_traffic_images, y: y_web})
    print(sess.run(tf.nn.top_k(tf.constant(top_k), k=3)))

# In[21]:


# ### Analyze Performance

# As we can see from results above we get an accuracy of 40%, correctly predicting the keep right and road work signs.

# ### Output Top 5 Softmax Probabilities For Each Image Found on the Web

# For each of the new images, print out the model's softmax probabilities to show the **certainty** of the model's predictions (limit the output to the top 5 probabilities for each image). [`tf.nn.top_k`](https://www.tensorflow.org/versions/r0.12/api_docs/python/nn.html#top_k) could prove helpful here. 
# 
# The example below demonstrates how tf.nn.top_k can be used to find the top k predictions for each image.
# 
# `tf.nn.top_k` will return the values and indices (class ids) of the top k predictions. So if k=3, for each sign, it'll return the 3 largest probabilities (out of a possible 43) and the correspoding class ids.
# 
# Take this numpy array as an example. The values in the array represent predictions. The array contains softmax probabilities for five candidate images with six possible classes. `tk.nn.top_k` is used to choose the three classes with the highest probability:
# 
# ```
# # (5, 6) array
# a = np.array([[ 0.24879643,  0.07032244,  0.12641572,  0.34763842,  0.07893497,
#          0.12789202],
#        [ 0.28086119,  0.27569815,  0.08594638,  0.0178669 ,  0.18063401,
#          0.15899337],
#        [ 0.26076848,  0.23664738,  0.08020603,  0.07001922,  0.1134371 ,
#          0.23892179],
#        [ 0.11943333,  0.29198961,  0.02605103,  0.26234032,  0.1351348 ,
#          0.16505091],
#        [ 0.09561176,  0.34396535,  0.0643941 ,  0.16240774,  0.24206137,
#          0.09155967]])
# ```
# 
# Running it through `sess.run(tf.nn.top_k(tf.constant(a), k=3))` produces:
# 
# ```
# TopKV2(values=array([[ 0.34763842,  0.24879643,  0.12789202],
#        [ 0.28086119,  0.27569815,  0.18063401],
#        [ 0.26076848,  0.23892179,  0.23664738],
#        [ 0.29198961,  0.26234032,  0.16505091],
#        [ 0.34396535,  0.24206137,  0.16240774]]), indices=array([[3, 0, 5],
#        [0, 1, 4],
#        [0, 5, 1],
#        [1, 3, 5],
#        [1, 4, 3]], dtype=int32))
# ```
# 
# Looking just at the first row we get `[ 0.34763842,  0.24879643,  0.12789202]`, you can confirm these are the 3 largest probabilities in `a`. You'll also notice `[3, 0, 5]` are the corresponding indices.

# The top five softmax probabilities for the predictions made above (code is in same cell where we calculated accuracy above)
# 
# TopKV2(values=array([[  1.00000000e+00,   0.00000000e+00,   0.00000000e+00],
#        [  9.31570530e-01,   6.56870678e-02,   2.74247653e-03],
#        [  1.00000000e+00,   0.00000000e+00,   0.00000000e+00],
#        [  6.96351230e-01,   1.56670287e-01,   1.46977261e-01],
#        [  8.00344706e-01,   1.99637681e-01,   1.31297438e-05]], dtype=float32), indices=array([[38,  0,  1],
#        [34,  7, 17],
#        [25,  0,  1],
#        [ 0,  1,  8],
#        [ 9, 23, 38]], dtype=int32))
# 

# ---
# 
# ## Step 4: Visualize the Neural Network's State with Test Images
# 
#  This Section is not required to complete but acts as an additional excersise for understaning the output of a neural network's weights. While neural networks can be a great learning device they are often referred to as a black box. We can understand what the weights of a neural network look like better by plotting their feature maps. After successfully training your neural network you can see what it's feature maps look like by plotting the output of the network's weight layers in response to a test stimuli image. From these plotted feature maps, it's possible to see what characteristics of an image the network finds interesting. For a sign, maybe the inner network feature maps react with high activation to the sign's boundary outline or to the contrast in the sign's painted symbol.
# 
#  Provided for you below is the function code that allows you to get the visualization output of any tensorflow weight layer you want. The inputs to the function should be a stimuli image, one used during training or a new one you provided, and then the tensorflow variable name that represents the layer's state during the training process, for instance if you wanted to see what the [LeNet lab's](https://classroom.udacity.com/nanodegrees/nd013/parts/fbf77062-5703-404e-b60c-95b78b2f3f9e/modules/6df7ae49-c61c-4bb2-a23e-6527e69209ec/lessons/601ae704-1035-4287-8b11-e2c2716217ad/concepts/d4aca031-508f-4e0b-b493-e7b706120f81) feature maps looked like for it's second convolutional layer you could enter conv2 as the tf_activation variable.
# 
# For an example of what feature map outputs look like, check out NVIDIA's results in their paper [End-to-End Deep Learning for Self-Driving Cars](https://devblogs.nvidia.com/parallelforall/deep-learning-self-driving-cars/) in the section Visualization of internal CNN State. NVIDIA was able to show that their network's inner weights had high activations to road boundary lines by comparing feature maps from an image with a clear path to one without. Try experimenting with a similar test to show that your trained network's weights are looking for interesting features, whether it's looking at differences in feature maps from images with or without a sign, or even what feature maps look like in a trained network vs a completely untrained one on the same sign image.
# 
# <figure>
#  <img src="visualize_cnn.png" width="380" alt="Combined Image" />
#  <figcaption>
#  <p></p> 
#  <p style="text-align: center;"> Your output should look something like this (above)</p> 
#  </figcaption>
# </figure>
#  <p></p> 
# 

# In[ ]:

### Visualize your network's feature maps here.
### Feel free to use as many code cells as needed.

# image_input: the test image being fed into the network to produce the feature maps
# tf_activation: should be a tf variable name used during your training procedure that represents the calculated state of a specific weight layer
# activation_min/max: can be used to view the activation contrast in more detail, by default matplot sets min and max to the actual min and max values of the output
# plt_num: used to plot out multiple different weight feature map sets on the same block, just extend the plt number for each new feature map entry

def outputFeatureMap(image_input, tf_activation, activation_min=-1, activation_max=-1 ,plt_num=1):
    # Here make sure to preprocess your image_input in a way your network expects
    # with size, normalization, ect if needed
    # image_input =
    # Note: x should be the same name as your network's tensorflow data placeholder variable
    # If you get an error tf_activation is not defined it maybe having trouble accessing the variable from inside a function
    activation = tf_activation.eval(session=sess,feed_dict={x : image_input})
    featuremaps = activation.shape[3]
    plt.figure(plt_num, figsize=(15,15))
    for featuremap in range(featuremaps):
        plt.subplot(6,8, featuremap+1) # sets the number of feature maps to show on each row and column
        plt.title('FeatureMap ' + str(featuremap)) # displays the feature map number
        if activation_min != -1 & activation_max != -1:
            plt.imshow(activation[0,:,:, featuremap], interpolation="nearest", vmin =activation_min, vmax=activation_max, cmap="gray")
        elif activation_max != -1:
            plt.imshow(activation[0,:,:, featuremap], interpolation="nearest", vmax=activation_max, cmap="gray")
        elif activation_min !=-1:
            plt.imshow(activation[0,:,:, featuremap], interpolation="nearest", vmin=activation_min, cmap="gray")
        else:
            plt.imshow(activation[0,:,:, featuremap], interpolation="nearest", cmap="gray")


# ### Question 9
# 
# Discuss how you used the visual output of your trained network's feature maps to show that it had learned to look for interesting characteristics in traffic sign images
# 

# **Answer:**

# > **Note**: Once you have completed all of the code implementations and successfully answered each question above, you may finalize your work by exporting the iPython Notebook as an HTML document. You can do this by using the menu above and navigating to  \n",
#     "**File -> Download as -> HTML (.html)**. Include the finished document along with this notebook as your submission.

# ### Project Writeup
# 
# Once you have completed the code implementation, document your results in a project writeup using this [template](https://github.com/udacity/CarND-Traffic-Sign-Classifier-Project/blob/master/writeup_template.md) as a guide. The writeup can be in a markdown or pdf file. 
