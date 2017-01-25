#importing some useful packages
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import numpy as np
import cv2

import math

def grayscale(img):
    """Applies the Grayscale transform
    This will return an image with only one color channel
    but NOTE: to see the returned image as grayscale
    (assuming your grayscaled image is called 'gray')
    you should call plt.imshow(gray, cmap='gray')"""
    return cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    # Or use BGR2GRAY if you read an image with cv2.imread()
    # return cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)


def canny(img, low_threshold, high_threshold):
    """Applies the Canny transform"""
    return cv2.Canny(img, low_threshold, high_threshold)


def gaussian_blur(img, kernel_size):
    """Applies a Gaussian Noise kernel"""
    return cv2.GaussianBlur(img, (kernel_size, kernel_size), 0)


def region_of_interest(img, vertices):
    """
    Applies an image mask.

    Only keeps the region of the image defined by the polygon
    formed from `vertices`. The rest of the image is set to black.
    """
    #defining a blank mask to start with
    mask = np.zeros_like(img)

    #defining a 3 channel or 1 channel color to fill the mask with depending on the input image
    if len(img.shape) > 2:
        channel_count = img.shape[2]  # i.e. 3 or 4 depending on your image
        ignore_mask_color = (255,) * channel_count
    else:
        ignore_mask_color = 255

    #filling pixels inside the polygon defined by "vertices" with the fill color
    cv2.fillPoly(mask, vertices, ignore_mask_color)

    #returning the image only where mask pixels are nonzero
    masked_image = cv2.bitwise_and(img, mask)
    return masked_image


# global vars (we are not using classes) to keep track of slopes and x coordinate values across calls to draw_lines
right_slope, left_slope, right_x0, right_x1, left_x0, left_x1, \
    last_right_line, last_left_line = None, None, None, None, None, None, None, None


def reset_global_draw_line_vars():
    global right_slope, left_slope, right_x0, right_x1, left_x0, left_x1, last_right_line, last_left_line
    right_slope, left_slope, right_x0, right_x1, left_x0, left_x1, \
        last_right_line, last_left_line = None, None, None, None, None, None, None, None


def low_pass_filter(x, y, alpha):
    """
    helper function to ensure smoother lines during transition between frames with help of global vars
    """
    return x*alpha + y*(1-alpha)


def draw_lines(img, lines, color=[0, 255, 0], thickness=5):
    imshape = img.shape
    global right_slope, left_slope, right_x0, right_x1, left_x0, left_x1
    right_slope_limits = (0.4, 0.9)
    left_slope_limits = (-0.4, -0.9)
    right_slope_points, left_slope_points = [], []
    for line in lines:
        for x1,y1,x2,y2 in line:
            slope = (y2 - y1)/(x2 - x1)
            if right_slope_limits[0] < slope < right_slope_limits[1]:
                right_slope_points.extend([(x1,y1), (x2,y2)])
            elif left_slope_limits[1] < slope < left_slope_limits[0]:
                left_slope_points.extend([(x1,y1), (x2,y2)])
            else:
                continue

    if right_slope_points and left_slope_points:
        current_right_slope_limits = (0.5, 0.8)
        current_left_slope_limits = (-0.5, -0.8)
        # find starting coordinates (i.e. coordinates in upper half of image)
        # for both lines having positive slope and negative slopes
        # choose minimum y coordinate point i.e. top most point
        right_starting_point = min(right_slope_points, key=lambda x: x[1])
        left_starting_point = min(left_slope_points, key=lambda x: x[1])

        # fit lines through the right points
        vx, vy, x, y = cv2.fitLine(np.array(right_slope_points), cv2.DIST_L2, 0, 0.01, 0.01)
        current_slope = vy/vx
        right_alpha = 0.3 if len(right_slope_points) > 5 else 0.8
        # check current slope is not noisy i.e. fall outside limits
        if current_right_slope_limits[0] < current_slope < current_right_slope_limits[1]:
            if right_slope is None:
                right_slope = current_slope
            else:
                right_slope = low_pass_filter(right_slope, current_slope, right_alpha)

            x0 = ((right_starting_point[1] - y)/right_slope) + x
            if right_x0 is None:
                right_x0 = x0
            else:
                right_x0 = low_pass_filter(right_x0, x0, right_alpha)

            x1 = math.ceil(((imshape[0] - y)/right_slope) + x)
            if right_x1 is None:
                right_x1 = x1
            else:
                right_x1 = low_pass_filter(right_x1, x1, right_alpha)
            cv2.line(img, (math.floor(right_x0), right_starting_point[1]), (math.floor(right_x1), imshape[0]), color, thickness)


        # repeat process for left points
        vx, vy, x, y = cv2.fitLine(np.array(left_slope_points), cv2.DIST_L2, 0, 0.01, 0.01)
        current_slope = vy/vx
        left_alpha = 0.3 if len(left_slope_points) > 5 else 0.8
        # check current slope is not noisy i.e. fall outside limits
        if current_left_slope_limits[1] < current_slope < current_left_slope_limits[0]:
            if left_slope is None:
                left_slope = current_slope
            else:
                left_slope = low_pass_filter(left_slope, current_slope, left_alpha)

            x0 = ((left_starting_point[1] - y)/left_slope) + x
            if left_x0 is None:
                left_x0 = x0
            else:
                left_x0 = low_pass_filter(left_x0, x0, left_alpha)

            x1 = math.ceil(((imshape[0] - y)/left_slope) + x)
            if left_x1 is None:
                left_x1 = x1
            else:
                left_x1 = low_pass_filter(left_x1, x1, left_alpha)

            cv2.line(img, (math.floor(left_x0), left_starting_point[1]), (math.floor(left_x1), imshape[0]), color, thickness)



def hough_lines(img, rho, theta, threshold, min_line_len, max_line_gap):
    """
    `img` should be the output of a Canny transform.

    Returns an image with hough lines drawn.
    """
    lines = cv2.HoughLinesP(img, rho, theta, threshold, np.array([]), minLineLength=min_line_len, maxLineGap=max_line_gap)
    line_img = np.zeros((img.shape[0], img.shape[1], 3), dtype=np.uint8)
    draw_lines(line_img, lines)
    return line_img

# Python 3 has support for cool math symbols.

def weighted_img(img, initial_img, α=0.8, β=1., λ=0.):
    """
    `img` is the output of the hough_lines(), An image with lines drawn on it.
    Should be a blank image (all black) with lines drawn on it.

    `initial_img` should be the image before any processing.

    The result image is computed as follows:

    initial_img * α + img * β + λ
    NOTE: initial_img and img must be the same shape!
    """
    return cv2.addWeighted(initial_img, α, img, β, λ)


def processing_pipeline(funcs, img, settings):
    for func in funcs:
        function_name = func.__name__
        args = settings.get(function_name)
        img = func(img, *args) if args else func(img)
        #plt.imshow(img)
        #plt.show()
    return img

import os
DIR_NAME = "test_images/"
test_images = os.listdir(DIR_NAME)
#test_images = ["solidWhiteRight.jpg"]
for image_file in test_images:
    print("Processing image file ", image_file)
    img = mpimg.imread(DIR_NAME + image_file)
    imshape = img.shape
    img_height, img_width = imshape[0], imshape[1]
    original_img = np.copy(img)
    funcs = [grayscale, gaussian_blur, canny, region_of_interest, hough_lines, weighted_img]
    function_settings = {
        'grayscale': [],
        'canny': [50, 150],
        'gaussian_blur': [5],
        'region_of_interest': [np.array([[(math.ceil(img_width/5), img_height), (math.ceil(img_width/4), math.ceil(0.6*img_height)),
                                          (math.ceil(4*img_width/5), math.ceil(0.6*img_height)), (img_width, img_height)]], dtype=np.int32)],
        'hough_lines': [2, np.pi/180, 40, 30, 200],
        'weighted_img': [original_img, 0.8, 1., 0.],
    }
    output_image = processing_pipeline(funcs, img, function_settings)
    # save output image
    #mpimg.imsave(DIR_NAME + "processed" + image_file[0].upper() + image_file[1:], output_image)
    plt.imshow(output_image)
    plt.show()
