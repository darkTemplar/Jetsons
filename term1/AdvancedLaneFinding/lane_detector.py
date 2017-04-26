import numpy as np
import cv2
import glob
import os
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
from skimage import img_as_ubyte
from moviepy.editor import VideoFileClip

### CONSTANTS
# Define conversions in x and y from pixels space to meters
YM_PER_PIX = 30/720 # meters per pixel in y dimension
XM_PER_PIX = 3.7/700 # meters per pixel in x dimension
MAX_DISCARDS = 3
# number of previous n lines we will be keeping track of (default 10)
MAX_LINES = 3
# minimum curvature in metres
MIN_CURVATURE = 1000
MIN_HORIZONTAL_DIST = 2000
MAX_SLOPE_DIFF = 10000

### VARIABLES
mtx = None
dist = None
M = None
Minv = None
current_left_fit, current_right_fit = None, None
left_slope, right_slope = None, None
#radius of curvature of the left and right lines in some units
left_radius_of_curvature, right_radius_of_curvature = None, None
#distance in meters of vehicle center from the line
line_base_pos = None
# store the last n lines which were good
last_n_lines_left, last_n_lines_right  = [], []
# number of line measurements which were discarded, will help us switch between full sliding window search and faster proximity search
num_discards = 0
use_fast_search = False
# boolean to indicate if intermediate results should be visualized
visualization = False


def calibrate_camera(folder_name='camera_cal', file_ext='.jpg', nx=9, ny=6):
    """

    :param folder_name: relative path to folder where calibration images are located
    :param file_ext: file extension for image files (defaults to .jpg)
    :param nx: number of rows on chessboard
    :param ny: number of columns
    :return: calibration matrix
    """
    global mtx, dist
    print("Starting camera calibration")
    # prepare object points, like (0,0,0), (1,0,0), (2,0,0) ....,(6,5,0)
    objp = np.zeros((ny*nx, 3), np.float32)
    objp[:, :2] = np.mgrid[0:nx, 0:ny].T.reshape(-1,2)

    # Arrays to store object points and image points from all the images.
    objpoints = [] # 3d point in real world space
    imgpoints = [] # 2d points in image plane.
    gray = None
    for fname in glob.glob(folder_name+'/*'+file_ext):
        img = mpimg.imread(fname)
        gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)

        # Find the chess board corners
        ret, corners = cv2.findChessboardCorners(gray, (nx, ny), None)

        # If found, add object points, image points (after refining them)
        if ret:
            objpoints.append(objp)
            imgpoints.append(corners)
            if visualization:
                # Draw and display the corners
                cv2.drawChessboardCorners(img, (nx, ny), corners, ret)
                cv2.imshow('img', img)
                cv2.waitKey()
    ret, mtx, dist, rvecs, tvecs = cv2.calibrateCamera(objpoints, imgpoints, gray.shape[::-1], None, None)


def undistort(img):
    undistorted = cv2.undistort(img, mtx, dist, None, mtx)
    if visualization:
        plt.imshow(undistorted)
        plt.show()
    return undistorted


def plot_image_comparison(before, after, effect_name='Undistorted'):
    f, (ax1, ax2) = plt.subplots(1, 2, figsize=(24, 9))
    f.tight_layout()
    ax1.imshow(before)
    ax1.set_title('Original Image', fontsize=50)
    ax2.imshow(after)
    ax2.set_title(effect_name + ' Image', fontsize=50)
    plt.subplots_adjust(left=0., right=1, top=0.9, bottom=0.)


def warp_image(img):
    global M, Minv
    if M is None or Minv is None:
        calibration_img = mpimg.imread('test_images/straight_lines1.jpg')
        # hardcode source points
        src_lower_left, src_lower_right, src_upper_right, src_upper_left = [249.823, 688.79], [1054.98, 688.79], [750.468, 490.081], [541.435, 490.081]
        src = np.float32([src_lower_left, src_lower_right, src_upper_right, src_upper_left])
        # estimate destination points based on src pt locations
        dst_lower_left, dst_lower_right, dst_upper_right, dst_upper_left = [350, 680], [940, 680], [940, 20], [350, 20]
        dst = np.float32([dst_lower_left, dst_lower_right, dst_upper_right, dst_upper_left])
        M = cv2.getPerspectiveTransform(src, dst)
        Minv = cv2.getPerspectiveTransform(dst, src)
        warped_calibration_img = cv2.warpPerspective(calibration_img, M, (calibration_img.shape[1], calibration_img.shape[0]), flags=cv2.INTER_LINEAR)
        """if visualization:
            plt.imshow(calibration_img)
            plt.plot(src_lower_left[0], src_lower_left[1], 'ro')
            plt.plot(src_lower_right[0], src_lower_right[1], 'bo')
            plt.plot(src_upper_right[0], src_upper_right[1], 'go')
            plt.plot(src_upper_left[0], src_upper_left[1], 'yo')
            plt.show()
            plt.imshow(warped_calibration_img)
            plt.plot(dst_lower_left[0], dst_lower_left[1], 'ro')
            plt.plot(dst_lower_right[0], dst_lower_right[1], 'bo')
            plt.plot(dst_upper_right[0], dst_upper_right[1], 'go')
            plt.plot(dst_upper_left[0], dst_upper_left[1], 'yo')
            plt.show()"""

    img_height, img_width = img.shape[:2]
    warped_img = cv2.warpPerspective(img, M, (img_width, img_height), flags=cv2.INTER_LINEAR)
    if visualization:
        plt.imshow(warped_img, cmap='gray')
        plt.show()
    return warped_img


def apply_gradient_and_color_thresholds(img, sobel_kernel=15, gradient_thresholds=(40, 100), color_thresholds=(170, 255)):
    # 1. apply color thresholds by only using S channel of HLS image
    hls = cv2.cvtColor(img, cv2.COLOR_RGB2HLS).astype(float)
    l_channel, s_channel = hls[:, :, 1], hls[:, :, 2]
    sbinary = np.zeros_like(s_channel)
    sbinary[(s_channel > color_thresholds[0]) & (s_channel <= color_thresholds[1])] = 1
    # 2. Apply gradient thresholds
    gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    sobelx = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=sobel_kernel)
    #sobely = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=sobel_kernel)
    abs_sobelx = np.absolute(sobelx)
    scaled_sobelx = np.uint8(255*abs_sobelx/np.max(abs_sobelx))
    sxbinary = np.zeros_like(scaled_sobelx)
    sxbinary[(scaled_sobelx > gradient_thresholds[0]) & (scaled_sobelx < gradient_thresholds[1])] = 1
    combined_binary = np.zeros_like(sxbinary)
    combined_binary[(sbinary == 1) | (sxbinary == 1)] = 1

    #visualization
    if visualization:
        color_binary = np.dstack((np.zeros_like(sxbinary), sxbinary, sbinary))
        # Plotting thresholded images
        f, (ax1, ax2) = plt.subplots(1, 2, figsize=(20, 10))
        ax1.set_title('Stacked thresholds')
        ax1.imshow(color_binary)
        ax2.set_title('Combined S channel and gradient thresholds')
        ax2.imshow(combined_binary, cmap='gray')
        plt.show()

    return combined_binary


def sanity_check(leftx, rightx):
    if left_radius_of_curvature < MIN_CURVATURE or right_radius_of_curvature < MIN_CURVATURE:
        print("Radius of Curvature failure")
        return False
    if abs(left_slope - right_slope) > 1:
        print("Lines lopes diverge more than allowed limit")
        return False
    if np.sum(rightx - leftx) < MIN_HORIZONTAL_DIST:
        print("Horizontal separation between lines less than limit")
        return False
    # all is good
    return True


def get_average_line(line_type='right'):
    """
    get average lane lines based on stored lane lines from previous frames
    :param line_type:
    :return:
    """
    last_n_lines = last_n_lines_right if line_type == 'right' else last_n_lines_left
    if not last_n_lines:
        return np.array([])
    # if less than the max_lines present then just return last good prediction
    if len(last_n_lines) < MAX_LINES:
        return last_n_lines[-1]
    num_points = 720
    line = last_n_lines[0]
    ploty = np.linspace(0, num_points, num_points)
    x = line[0]*ploty**2 + line[1]*ploty + line[2]
    y = ploty
    for new_line in last_n_lines[1:]:
        y = np.append(y, ploty)
        fitx = new_line[0]*ploty**2 + new_line[1]*ploty + new_line[2]
        x = np.append(x, fitx)
    return np.polyfit(y, x, 2)


def update_discards():
    """
    update number of frame calculations discarded and decide if we should switch away from fast search
    :return:
    """
    global num_discards, use_fast_search
    num_discards += 1
    # if max discards reached, go back to full window search and reset discards
    if num_discards > MAX_DISCARDS:
        num_discards = 0
        use_fast_search = False


def sliding_window_search(warped_img, windows=9, margin=50, min_pixels=100):
    """
    Use intensity histogram and full window search to figure out lane lines in the image
    :param warped_img:
    :param windows: number of windows to be searched in
    :param margin: extra margin to be searched around left and right centroids
    :param min_pixels: min. number of pixels to be found to re-center
    :return:
    """
    global current_left_fit, current_right_fit, use_fast_search, last_n_lines_left, last_n_lines_right
    img_height, img_width = warped_img.shape[:2]
    # Take a pixel intensities histogram of the bottom half of the image
    histogram = np.sum(warped_img[img_height//2:, :], axis=0)
    if visualization:
        # Create an output image to draw on and  visualize the result
        out_img = np.dstack((warped_img, warped_img, warped_img))*255
    # Find the peak of the left and right halves of the histogram
    # These will be the starting point for the left and right lines
    midpoint = np.int(histogram.shape[0]//2)
    leftx_base, rightx_base = np.argmax(histogram[:midpoint]), np.argmax(histogram[midpoint:]) + midpoint

    # Set height of windows
    window_height = img_height//windows
    # Identify the x and y positions of all nonzero pixels in the image
    nonzero = warped_img.nonzero()
    nonzeroy, nonzerox = np.array(nonzero[0]), np.array(nonzero[1])
    # Current positions to be updated for each window
    leftx_current, rightx_current = leftx_base, rightx_base
    # Create empty lists to receive left and right lane pixel indices
    left_indexes, right_indexes = [], []

    # Step through the windows one by one
    for window in range(windows):
        # Identify window boundaries in x and y (and right and left)
        win_y_low = img_height - (window+1)*window_height
        win_y_high = img_height - window*window_height
        win_xleft_low = leftx_current - margin
        win_xleft_high = leftx_current + margin
        win_xright_low = rightx_current - margin
        win_xright_high = rightx_current + margin
        if visualization:
            # Draw the windows on the visualization image
            cv2.rectangle(out_img,(win_xleft_low,win_y_low),(win_xleft_high,win_y_high),(0,255,0), 2)
            cv2.rectangle(out_img,(win_xright_low,win_y_low),(win_xright_high,win_y_high),(0,255,0), 2)
        # Identify the nonzero pixels in x and y within the window
        good_left_indexes = ((nonzeroy >= win_y_low) & (nonzeroy < win_y_high) & (nonzerox >= win_xleft_low) &
                          (nonzerox < win_xleft_high)).nonzero()[0]
        good_right_indexes = ((nonzeroy >= win_y_low) & (nonzeroy < win_y_high) & (nonzerox >= win_xright_low) &
                           (nonzerox < win_xright_high)).nonzero()[0]
        # Append these indices to the lists
        #print("Number of good indices in window ", window)
        #print(len(good_left_indexes))
        #print(len(good_right_indexes))
        # If you found > minpix pixels, recenter next window on their mean position
        if len(good_left_indexes) > min_pixels:
            leftx_current = np.int(np.mean(nonzerox[good_left_indexes]))
            left_indexes.append(good_left_indexes)
        if len(good_right_indexes) > min_pixels:
            rightx_current = np.int(np.mean(nonzerox[good_right_indexes]))
            right_indexes.append(good_right_indexes)

    # Concatenate the arrays of indices
    if left_indexes:
        left_indexes = np.concatenate(left_indexes)
        # Extract left and right line pixel positions
        leftx = nonzerox[left_indexes]
        lefty = nonzeroy[left_indexes]
        # Fit a second order polynomial to each
        current_left_fit = np.polyfit(lefty, leftx, 2)

    # repeat for right
    if right_indexes:
        right_indexes = np.concatenate(right_indexes)
        # Extract left and right line pixel positions
        rightx = nonzerox[right_indexes]
        righty = nonzeroy[right_indexes]
        current_right_fit = np.polyfit(righty, rightx, 2)

    if visualization:
        # Generate x and y values for plotting
        ploty = np.linspace(0, warped_img.shape[0]-1, warped_img.shape[0])
        left_fitx = current_left_fit[0]*ploty**2 + current_left_fit[1]*ploty + current_left_fit[2]
        right_fitx = current_right_fit[0]*ploty**2 + current_right_fit[1]*ploty + current_right_fit[2]

        out_img[nonzeroy[left_indexes], nonzerox[left_indexes]] = [255, 0, 0]
        out_img[nonzeroy[right_indexes], nonzerox[right_indexes]] = [0, 0, 255]
        plt.imshow(out_img)
        plt.plot(left_fitx, ploty, color='yellow')
        plt.plot(right_fitx, ploty, color='yellow')
        plt.xlim(0, 1280)
        plt.ylim(720, 0)
        plt.show()


def fast_window_search(warped_img, margin=100):
    """
    Use previous fitted lane lines to quickly approximate lane lines in the current image
    :param warped_img:
    :param margin:
    :param visualization:
    :return:
    """
    global current_left_fit, current_right_fit
    if visualization:
        # Create an output image to draw on and  visualize the result
        out_img = np.dstack((warped_img, warped_img, warped_img))*255
    # Assume you now have a new warped binary image
    # It's now much easier to find line pixels!
    nonzero = warped_img.nonzero()
    nonzeroy = np.array(nonzero[0])
    nonzerox = np.array(nonzero[1])
    left_lane_inds = ((nonzerox > (current_left_fit[0]*(nonzeroy**2) + current_left_fit[1]*nonzeroy + current_left_fit[2] - margin)) & (nonzerox < (current_left_fit[0]*(nonzeroy**2) + current_left_fit[1]*nonzeroy + current_left_fit[2] + margin)))
    right_lane_inds = ((nonzerox > (current_right_fit[0]*(nonzeroy**2) + current_right_fit[1]*nonzeroy + current_right_fit[2] - margin)) & (nonzerox < (current_right_fit[0]*(nonzeroy**2) + current_right_fit[1]*nonzeroy + current_right_fit[2] + margin)))

    # Again, extract left and right line pixel positions
    leftx = nonzerox[left_lane_inds]
    lefty = nonzeroy[left_lane_inds]
    rightx = nonzerox[right_lane_inds]
    righty = nonzeroy[right_lane_inds]
    # Fit a second order polynomial to each
    left_fit, right_fit = np.polyfit(lefty, leftx, 2), np.polyfit(righty, rightx, 2)
    current_left_fit, current_right_fit = left_fit, right_fit

    if visualization:
        # Generate x and y values for plotting
        ploty = np.linspace(0, warped_img.shape[0]-1, warped_img.shape[0])
        left_fitx = current_left_fit[0]*ploty**2 + current_left_fit[1]*ploty + current_left_fit[2]
        right_fitx = current_right_fit[0]*ploty**2 + current_right_fit[1]*ploty + current_right_fit[2]
        out_img[lefty, leftx] = [255, 0, 0]
        out_img[righty, rightx] = [0, 0, 255]
        plt.imshow(out_img)
        plt.plot(left_fitx, ploty, color='yellow')
        plt.plot(right_fitx, ploty, color='yellow')
        plt.xlim(0, 1280)
        plt.ylim(720, 0)
        plt.show()


def finalize_params():
    """
    Run the calculated params like radius of curvature and lane lines via a sanity check
    :return:
    """
    global current_left_fit, current_right_fit, use_fast_search, last_n_lines_left, last_n_lines_right
    leftx_cr, rightx_cr = radius_of_curvature()
    if not sanity_check(leftx_cr, rightx_cr):
        print("Discarding frame and resetting params")
        update_discards()
        # drop current line fits and try to get average lines instead
        left_fit, right_fit = get_average_line(line_type='left'), get_average_line()
        if left_fit.size and right_fit.size:
            radius_of_curvature()
            current_left_fit, current_right_fit = left_fit, right_fit

    else:
        use_fast_search = True
        # store good lane line fits for future average calculations
        last_n_lines_left.append(current_left_fit)
        if len(last_n_lines_left) > MAX_LINES:
            last_n_lines_left = last_n_lines_left[1:]
        last_n_lines_right.append(current_right_fit)
        if len(last_n_lines_right) > MAX_LINES:
            last_n_lines_right = last_n_lines_right[1:]


def radius_of_curvature():
    """
    Calculate left and right radius of curvature in metres
    :return: left and right lane lines (in metres)
    """
    global left_radius_of_curvature, right_radius_of_curvature, left_slope, right_slope
    ploty = np.linspace(0, 719, num=720)# to cover same y-range as image
    leftx = np.array([(y**2)*current_left_fit[0] + current_left_fit[1] * y + current_left_fit[2] for y in ploty])
    rightx = np.array([(y**2)*current_right_fit[0] + current_right_fit[1] * y + current_right_fit[2] for y in ploty])
    leftx, rightx = leftx[::-1], rightx[::-1]  # Reverse to match top-to-bottom in y
    y_eval = np.max(ploty)
    # Fit new polynomials to x,y in world space
    left_fit_cr = np.polyfit(ploty*YM_PER_PIX, leftx*XM_PER_PIX, 2)
    right_fit_cr = np.polyfit(ploty*YM_PER_PIX, rightx*XM_PER_PIX, 2)
    # Calculate the new radii of curvature
    left_slope, right_slope = 2*left_fit_cr[0]*y_eval*YM_PER_PIX + left_fit_cr[1], 2*right_fit_cr[0]*y_eval*YM_PER_PIX + right_fit_cr[1]
    #print("left slope", left_slope)
    #print("right slope", right_slope)
    left_radius_of_curvature = ((1 + left_slope**2)**1.5) / np.absolute(2*left_fit_cr[0])
    right_radius_of_curvature = ((1 + right_slope**2)**1.5) / np.absolute(2*right_fit_cr[0])
    #print("left radius", left_radius_of_curvature)
    #print("right radius", right_radius_of_curvature)
    return leftx*XM_PER_PIX, rightx*XM_PER_PIX


def draw_on_original(original_img, warped_img):
    """
    draw colored lane lines and other metadat like radius of curvature on the final image
    :param original_img:
    :param warped_img:
    :return:
    """
    # Create an image to draw the lines on
    warp_zero = np.zeros_like(warped_img)
    color_warp = np.dstack((warp_zero, warp_zero, warp_zero))
    # Generate x and y values for plotting
    ploty = np.linspace(0, warped_img.shape[0]-1, warped_img.shape[0])
    left_fitx = current_left_fit[0]*ploty**2 + current_left_fit[1]*ploty + current_left_fit[2]
    right_fitx = current_right_fit[0]*ploty**2 + current_right_fit[1]*ploty + current_right_fit[2]
    # Recast the x and y points into usable format for cv2.fillPoly()
    pts_left = np.array([np.transpose(np.vstack([left_fitx, ploty]))])
    pts_right = np.array([np.flipud(np.transpose(np.vstack([right_fitx, ploty])))])
    pts = np.hstack((pts_left, pts_right))

    # Draw the lane onto the warped blank image
    cv2.fillPoly(color_warp, np.int_([pts]), (0, 255, 0))

    # Warp the blank back to original image space using inverse perspective matrix (Minv)
    unwarped_img = cv2.warpPerspective(color_warp, Minv, (original_img.shape[1], original_img.shape[0]))
    # Combine the result with the original image
    result = cv2.addWeighted(original_img, 1, unwarped_img, 0.3, 0)
    # TODO: write offset of central camera (car center) from lane center (use cv2.putText)
    cv2.putText(result, "Left Curvature: " + str(left_radius_of_curvature) + " (m)", (100, 100), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), thickness=4)
    cv2.putText(result, "Right Curvature: " + str(right_radius_of_curvature) + " (m)", (300, 300), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), thickness=4)
    #cv2.putText(result, "Distance from center: " + str(dist_from_center) + " (m)", (100, 150), cv2.FONT_HERSHEY_SIMPLEX, 2, (255,255,255))
    if visualization:
        plt.imshow(result)
        plt.show()
    return result


def run_pipeline(img, draw_images=False):
    """
    collection of functions to call on an image to produce lane lines and calculate radius of curvature
    :param img:
    :param draw_images:
    :return:
    """
    global visualization
    # calibrate camera during initialization
    if mtx is None and dist is None:
        calibrate_camera()
    if draw_images:
        visualization = True

    # 1. Undistort image
    undistorted_img = undistort(img)
    # 2. Apply color and gradient thresholds
    binary_thresholded_image = apply_gradient_and_color_thresholds(undistorted_img)
    # 3. warp image
    binary_warped_image = warp_image(binary_thresholded_image)
    # 4. fit lines using sliding window search
    if use_fast_search:
        fast_window_search(binary_warped_image)
    else:
        sliding_window_search(binary_warped_image)
    # 5. Calculate radius of curvature and reset fit lines and radius if needed
    finalize_params()
    # 6. Draw detected lane lines and radius of curvature back on original image
    result = draw_on_original(img, binary_warped_image)
    return result


def process_images(folder_name, file_ext='.jpg'):
    """
    process images in given folder via pipeline. Used to test pipeline on individual images
    :param folder_name:
    :param file_ext:
    :return:
    """
    for fname in glob.glob(folder_name+'/*'+file_ext):
        # 1. read image
        print("processing", fname)
        img = mpimg.imread(fname)
        if img.dtype != 'uint8':
            img = img_as_ubyte(img)
        plt.imshow(img)
        plt.show()
        run_pipeline(img, True)


def process_video(video_file, output_file):
    """
    Process input video and produce output video with lane lines and curvature drawn on it
    :param video_file:
    :param output_file:
    :return:
    """
    clip1 = VideoFileClip(video_file)
    white_clip = clip1.fl_image(run_pipeline) #NOTE: this function expects color images!!
    white_clip.write_videofile(output_file, audio=False)


def extract_frames(video_file, times, folder_name='more_test'):
    """
    Extract images from various times (in sec) from the video for testing
    :param video_file: absolute path to input file
    :param times: list of times for which image snpashots are taken
    :param folder_name: folder where images are to be saved
    :return:
    """
    clip = VideoFileClip(video_file)
    for t in times:
        imgpath = os.path.join(folder_name, '{}.png'.format(t))
        clip.save_frame(imgpath, t, withmask=False)



#run_pipeline(mpimg.imread('test_images/straight_lines1.jpg'), True)
#process_images('test_images')
#process_images('project_test', '.png')
#process_video('project_video.mp4', 'project_output.mp4')
process_video('challenge_video.mp4', 'challenge_output.mp4')
#extract_frames('challenge_video.mp4', [1, 2, 3, 4, 5, 6, 7])