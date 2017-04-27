---

**Advanced Lane Finding Project**

The goals / steps of this project are the following:

* Compute the camera calibration matrix and distortion coefficients given a set of chessboard images.
* Apply a distortion correction to raw images.
* Use color transforms, gradients, etc., to create a thresholded binary image.
* Apply a perspective transform to rectify binary image ("birds-eye view").
* Detect lane pixels and fit to find the lane boundary.
* Determine the curvature of the lane and vehicle position with respect to center.
* Warp the detected lane boundaries back onto the original image.
* Output visual display of the lane boundaries and numerical estimation of lane curvature and vehicle position.

[//]: # (Image References)

[image1]: ./examples/undistort_output.png "Undistorted"
[image2]: ./test_images/test1.jpg "Road Transformed"
[image3]: ./examples/binary_combo_example.jpg "Binary Example"
[image4]: ./examples/warped_straight_lines.jpg "Warp Example"
[image5]: ./examples/color_fit_lines.jpg "Fit Visual"
[image6]: ./examples/example_output.jpg "Output"
[video1]: ./project_video.mp4 "Video"

## [Rubric](https://review.udacity.com/#!/rubrics/571/view) Points
###Here I will consider the rubric points individually and describe how I addressed each point in my implementation.  

---
###Writeup / README

####1. Provide a Writeup / README that includes all the rubric points and how you addressed each one.  You can submit your writeup as markdown or pdf.  [Here](https://github.com/udacity/CarND-Advanced-Lane-Lines/blob/master/writeup_template.md) is a template writeup for this project you can use as a guide and a starting point.  

You're reading it! Please note all function references and line numbers will refer to file lane_detector.py
###Camera Calibration

####1. Briefly state how you computed the camera matrix and distortion coefficients. Provide an example of a distortion corrected calibration image.

The code for this step is contained in the `calibrate_camera` function (line 42)

I start by preparing "object points", which will be the (x, y, z) coordinates of the chessboard corners in the world. Here I am assuming the chessboard is fixed on the (x, y) plane at z=0, such that the object points are the same for each calibration image.  Thus, `objp` is just a replicated array of coordinates, and `objpoints` will be appended with a copy of it every time I successfully detect all chessboard corners in a test image.  `imgpoints` will be appended with the (x, y) pixel position of each of the corners in the image plane with each successful chessboard detection. For this task, I used the images provided in the camera_cal folder which had 9x6 chessboard images. 

I then used the output `objpoints` and `imgpoints` to compute the camera calibration and distortion coefficients using the `cv2.calibrateCamera()` function.  I applied this distortion correction to the test image using the `cv2.undistort()` function and obtained this result: 

![alt text][original_chessboard]
![alt text][undistorted_chessboard]

###Pipeline (single images)

####1. Provide an example of a distortion-corrected image.
To demonstrate this step, I will describe how I apply the distortion correction to one of the test images like this one:
![alt text][original_test_image]
####2. Describe how (and identify where in your code) you used color transforms, gradients or other methods to create a thresholded binary image.  Provide an example of a binary image result.
I used a combination of color and gradient thresholds to generate a binary image. This is detailed in the function `apply_gradient_and_color_thresholds` on line 133. For color thresholding I converted the image to HLS color space and then extracted the S (saturation) channel since it seemed the most robust to shadows and other noisy data in the images. For gradient thresholding I experimented with magnitude, direction and absolute gradient approaches. Finally chose absolute gradient as that combined with color thresholding seemed to give the best results for the various test images. Here's an example of my output for this step.

![alt text][binary_img]

####3. Describe how (and identify where in your code) you performed a perspective transform and provide an example of a transformed image.

The code for my perspective transform is contained in the function `warp_image` on line 98.  The function takes as inputs an image (`img`) and returns the warped version of the image. It may also calculate the variables M (Perspective transform matrix) and Minv (Inverse Perspective Transform Matrix) if they have not been previously calculated. To calculate those matrices, I first tried using edge and corner detection to programmatically detect the source and destination points, however that approach did not seem to generalize well compared to manually inspecting a test image and then hardcoding the source and destination points. This is probably an area of improvement for the project in the future. The manual hardcoding yielded the following source and destination points (order is lower left, lower right, upper right, upper left)

| Source         | Destination   | 
|:-------------: |:-------------:| 
|249.823,688.79  | 350, 680      | 
|1054.98, 688.79 | 940, 680      |
|750.468, 490.081| 940, 20       |
|541.435, 490.081| 350, 20       |

I verified that my perspective transform was working as expected by drawing the `src` and `dst` points onto a test image and its warped counterpart to verify that the lines appear parallel in the warped image.

![alt text][warped_img]

####4. Describe how (and identify where in your code) you identified lane-line pixels and fit their positions with a polynomial? Also describe how (and identify where in your code) you calculated the radius of curvature of the lane and the position of the vehicle with respect to center.

To find the lines in my image, I used 2 main functions. `sliding_window_search` line 223 and `fast_window_search` line 317. The `sliding_window_search` took the warped binary thresholded image as input and then drew a histogram of the pixel intensities in the lower half of the image. The basic assumption here being that peaks in the left and right halves of the histogram would identify the base of the left and right lane lines. Once we have the base of the lines, we divide the left and right halves of the image into windows (function accepts this as a param). The center of the first (bottom) window is the base we found via the histogram. The width of the windows is controlled by a margin (function accepts this as param as well). We then search for white pixels in this window. If we find enough pixels, then we re-center our base for the next window. Via experimentation, I found that keeping the margins relatively narrow protects against outlier pixels which can corrupt the polynomial fitting. I also have a minimum pixel threshold for a window before we accept pixels detected in that window, this provides for higher confidence measurements.

We then use the fitted lines to calculate the radius of curvature using the function `radius_of_curvature` on line 388. We convert values from the image space (pixels) into real world space (metres). 

If the fitted lines and radius of curvature values pass the sanity checks detailed in function `sanity_check` on line 172 (min radius of curvature, min horizontal distance and nearly parellel requirements), we then switch to using the `fast_window_search` method (controlled by flag `use_fast_search`) for finding lane lines for the next frame. In this method we use the lines found in the previous frame and search for the white pixels in an area (controlled by margin param in the function) around the lines. This allows for much faster lane line detections. However if the sanity check fails, then we update the discard count and if it reaches a certain threshold, then we set `use_fast_search` to false and go back to `sliding_window_search`. For the lines, we try and get an average of the 3 previous high confidence measurements, which we had been preserving. This entire logic is contained in function `finalize_params` on line 361.

![alt text][lane_lines_img]


####6. Provide an example image of your result plotted back down onto the road such that the lane area is identified clearly.

I implemented this step in function `draw_on_original`  Here is an example of my result on a test image:

![alt text][final_img]

---

###Pipeline (video)

####1. Provide a link to your final video output.  Your pipeline should perform reasonably well on the entire project video (wobbly lines are ok but no catastrophic failures that would cause the car to drive off the road!).

Here's a [link to my video result](./project_output.mp4)

---

###Discussion

####1. Briefly discuss any problems / issues you faced in your implementation of this project.  Where will your pipeline likely fail?  What could you do to make it more robust?

Here I'll talk about the approach I took, what techniques I used, what worked and why, where the pipeline might fail and how I might improve it if I were going to pursue this project further.  

