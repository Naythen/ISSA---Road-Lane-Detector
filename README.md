# ISSA---Road-Lane-Detector
A road lane detector 


# 1. Import necessary libraries and start code
import cv2
import numpy as np

# Initialize the video we want to use
cam = cv2.VideoCapture('Lane Detection Test Video-01.mp4')

# global variables so I don't get random errors
l_bs = [0, 0]
l_as = [0, 0]
r_bs = [0, 0]
r_as = [0, 0]
final_l_bs = [0, 0]
final_l_as = [0, 0]
final_r_bs = [0, 0]
final_r_as = [0, 0]
left_top = [0, 0]
right_top = [0, 0]
left_bottom = [0, 0]
right_bottom = [0, 0]
final_left_top = [0, 0]
final_right_top = [0, 0]
final_left_bottom = [0, 0]
final_right_bottom = [0, 0]
final_left_top_x = 0
final_left_bottom_x = 0
final_right_top_x = 0
final_right_bottom_x = 0

# Infinite loop for processing each frame
while True:
    # Reading frame by frame the entire video
    ret, frame = cam.read()
    if ret is False:
        break

    # cv2.imshow('Original', frame)
    # print(frame.shape)

    # 2. Shrink the frame!
    scale_percent = 30  # procentul pe care il pastram din frame-ul original
    scale_factor = scale_percent / 100  # factor de scalare a imaginii

    # Dimensiunile frame-ului resized
    frame_height = int(frame.shape[0] * scale_factor)
    frame_width = int(frame.shape[1] * scale_factor)
    frame_dimensions = (frame_width, frame_height)
    # print(frame_dimensions)

    small = cv2.resize(frame, frame_dimensions)
    cv2.imshow('Small', small)
    # print("Small:",small.shape)

    # 3. Convert the frame to Grayscale!
    gray_frame = cv2.cvtColor(small, cv2.COLOR_BGR2GRAY)
    cv2.imshow('Grayscale', gray_frame)
    # print("Gray:",gray_frame.shape)

    # 4. Select only the road!
    new_frame = np.zeros([frame_height, frame_width], dtype=np.uint8)  # creez un frame nou
    # trapez = cv2.resize(trapez,frame_dimensions)
    # cv2.imshow('Trapezoid', new_frame)

    # coordonate colturi trapez
    trapez_upper_left = (int(frame_width * 0.45), int(frame_height * 0.80))
    trapez_upper_right = (int(frame_width * 0.55), int(frame_height * 0.80))
    trapez_lower_left = (int(frame_width * 0), int((frame_height * 1.0)))
    trapez_lower_right = (int(frame_width * 1.0), int((frame_height * 1.0)))

    # trapez_bounds = [upper_right, upper_left, lower_left, lower_right]
    trapez_bounds = np.array([trapez_upper_left, trapez_upper_right, trapez_lower_right, trapez_lower_left],
                             dtype=np.int32)
    trapez_frame = cv2.fillConvexPoly(new_frame, trapez_bounds, 255)
    cv2.imshow('Trapezoid', trapez_frame)
    # print("Trapez:", trapez_frame.shape)

    road = trapez_frame * gray_frame
    # road = cv2.bitwise_not(road)
    # print(road.shape)
    cv2.imshow('Road', road * 255)

    # 5. Get a top-down view!
    # coordonate colturi frame resized intreg
    frame_upper_left = (0, 0)
    frame_upper_right = (frame_width, 0)
    frame_lower_left = (0, frame_height)
    frame_lower_right = (frame_width, frame_height)

    frame_bounds = np.array([frame_upper_left, frame_upper_right, frame_lower_right, frame_lower_left],
                            dtype=np.float32)
    trapez_bounds = np.float32(trapez_bounds)

    magic_matrix = cv2.getPerspectiveTransform(trapez_bounds, frame_bounds)
    top_down_frame = cv2.warpPerspective(road, magic_matrix, frame_dimensions)
    cv2.imshow('Top-Down', top_down_frame * 255)

    # 6. Adding a bit of blur!
    ksize = (5, 5)
    blurred_frame = cv2.blur(top_down_frame, ksize)
    cv2.imshow('Blur', blurred_frame * 255)

    # 7. Edge detection!
    # a. Create vertical sobel
    sobel_vertical = np.float32([[-1, -2, -1], [0, 0, 0], [+1, +2, +1]])
    # b. Create horizontal sobel
    sobel_horizontal = np.transpose(sobel_vertical)

    blurred_frame = np.float32(blurred_frame)

    sobel_vertical_frame = cv2.filter2D(blurred_frame, -1, sobel_vertical)
    sobel_horizontal_frame = cv2.filter2D(blurred_frame, -1, sobel_horizontal)

    sobel_final = np.sqrt((sobel_vertical_frame*sobel_vertical_frame)+(sobel_horizontal_frame*sobel_horizontal_frame))
    sobel_vertical_frame = cv2.convertScaleAbs(sobel_vertical_frame)
    sobel_horizontal_frame = cv2.convertScaleAbs(sobel_horizontal_frame)
    sobel_final_frame = cv2.convertScaleAbs(sobel_final)

    cv2.imshow('Vertical Sobel', sobel_vertical_frame)
    cv2.imshow('Horizontal Sobel', sobel_horizontal_frame)
    cv2.imshow('Final Sobel', sobel_final_frame)

    # 8. Binarize the frame!
    threshold = 255 / 2
    ret, binarized_frame = cv2.threshold(sobel_final, threshold, 255, cv2.THRESH_BINARY)
    # binarized_frame_final = cv2.convertScaleAbs(binarized_frame)
    cv2.imshow('Binarized Frame', binarized_frame)

    # 9. Lines!
    # a. removing some noise
    copy_frame = binarized_frame.copy()
    copy_frame[:, 0:int(0.05 * frame_width)] = 0
    copy_frame[:, int(0.95 * frame_width): int(1.0 * frame_width)] = 0
    cv2.imshow('9th Frame', copy_frame)

    # b. Coordinates of white points in left and right halves
    left_white_pixels = np.argwhere(copy_frame[:, :int(0.5 * frame_width)] > 255//2)
    right_white_pixels = np.argwhere(copy_frame[:, int(0.5 * frame_width):] > 255//2)

    left_xs = left_white_pixels[:, 1]
    left_ys = left_white_pixels[:, 0]
    right_xs = right_white_pixels[:, 1] + frame_width//2
    right_ys = right_white_pixels[:, 0]

    # 10. Find the lines that detect the edges of the lane!
    # a.
    if len(left_xs) != 0 or len(left_ys) != 0:
        # If points exist then we calculate b and a
        l_bs, l_as = np.polynomial.polynomial.polyfit(left_xs, left_ys, deg=1)

    if len(right_xs) != 0 or len(right_ys) != 0:
        # same as before
        r_bs, r_as = np.polynomial.polynomial.polyfit(right_xs, right_ys, deg=1)

    # y = ax + b
    # Calculate coordinates of the top and bottom points of the left and right lanes
    left_top_y = 0
    left_top_x = (left_top_y-l_bs)/l_as

    left_bottom_y = frame_height
    left_bottom_x = (left_bottom_y-l_bs)/l_as

    right_top_y = 0
    right_top_x = (right_top_y-r_bs)/r_as

    right_bottom_y = frame_height
    right_bottom_x = (right_bottom_y-r_bs)/r_as

    left_top[1] = left_top_y
    left_bottom[1] = left_bottom_y
    right_top[1] = right_top_y
    right_bottom[1] = right_bottom_y

    # if -frame_width/2 < left_top_x < frame_width/2:
    left_top[0] = int(left_top_x)
    # if -frame_width/2 < left_bottom_x < frame_width/2:
    left_bottom[0] = int(left_bottom_x)

    # if frame_width/2 < right_top_x < 3/2 * frame_width:
    right_top[0] = int(right_top_x)  # + frame_width//2
    # if frame_width/2 < right_bottom_x < 3/2 * frame_width:
    right_bottom[0] = int(right_bottom_x)  # + frame_width//2

    # Draw lines on the binarized frame
    frame_lines = cv2.line(binarized_frame, left_top, left_bottom, (200, 0, 0), 5)
    frame_lines = cv2.line(frame_lines, right_top, right_bottom, (100, 0, 0), 5)
    frame_lines = cv2.line(frame_lines, (int(frame_width/2), 0), (int(frame_width/2), int(frame_height)), (255, 0, 0),
                           1)
    # cv2.line(frame_with_lines, left_top, left_bottom, (255, 0, 0), 5)
    # cv2.line(frame_with_lines, right_top, right_bottom, (0, 255, 0), 5)

    cv2.imshow('Lines', frame_lines)

    # 11. Transform lines back to the original perspective
    new_left_frame = np.zeros((frame_dimensions[1], frame_dimensions[0]), dtype=np.uint8)
    new_left_frame = cv2.line(new_left_frame, left_top, left_bottom, (255, 0, 0), 5)
    cv2.imshow('alt test left lane', new_left_frame)
    new_magic_matrix = cv2.getPerspectiveTransform(frame_bounds, trapez_bounds)
    back_to_normal_frame = cv2.warpPerspective(new_left_frame, new_magic_matrix, frame_dimensions)
    cv2.imshow('Test1', back_to_normal_frame)

    new_right_frame = np.zeros((frame_dimensions[1], frame_dimensions[0]), dtype=np.uint8)

    new_right_frame = cv2.line(new_right_frame, right_top, right_bottom, (255, 0, 0), 5)
    cv2.imshow('alt test', new_right_frame)
    # new_magic_matrix = cv2.getPerspectiveTransform(frame_bounds, trapez_bounds)
    back_to_normal_right_frame = cv2.warpPerspective(new_right_frame, new_magic_matrix, frame_dimensions)
    cv2.imshow('Test2', back_to_normal_right_frame)

    # Find white pixels in the transformed frames
    final_left_white_pixels = np.argwhere(back_to_normal_frame > 0)
    final_right_white_pixels = np.argwhere(back_to_normal_right_frame > 0)

    # Fit lines to the white pixels
    final_left_xs = final_left_white_pixels[:, 1]
    final_left_ys = final_left_white_pixels[:, 0]

    final_right_xs = final_right_white_pixels[:, 1]  # + frame_width // 2
    final_right_ys = final_right_white_pixels[:, 0]

    if len(final_left_xs) != 0 or len(final_left_ys) != 0:
        # If points exist then calculate
        final_l_bs, final_l_as = np.polynomial.polynomial.polyfit(final_left_xs, final_left_ys, deg=1)

    if len(final_right_xs) != 0 or len(final_right_ys) != 0:
        # same thing as above
        final_r_bs, final_r_as = np.polynomial.polynomial.polyfit(final_right_xs, final_right_ys, deg=1)

    # print("final_l_bs:",final_l_bs,"final_l_as:",final_l_as)
    # Calculating coordinates of the top and bottom points of the final left and right lanes
    final_left_top_y = int(frame_height * 0.8)
    if not (abs(int((final_left_top_y - final_l_bs) / final_l_as)) > 10**7):
        final_left_top_x = int((final_left_top_y - final_l_bs) / final_l_as)

    final_left_bottom_y = int(frame_height)
    if not (abs(int((final_left_bottom_y - final_l_bs) / final_l_as)) > 10 ** 7):
        final_left_bottom_x = int((final_left_bottom_y - final_l_bs) / final_l_as)

    final_right_top_y = int(frame_height * 0.8)
    if not (abs(int((final_right_top_y - final_r_bs) / final_r_as)) > 10**7):
        final_right_top_x = int((final_right_top_y - final_r_bs) / final_r_as)

    final_right_bottom_y = int(frame_height)
    if not (abs(int((final_right_bottom_y - final_r_bs) / final_r_as)) > 10 ** 7):
        final_right_bottom_x = int((final_right_bottom_y - final_r_bs) / final_r_as)

    final_left_top[1] = int(final_left_top_y)
    final_left_bottom[1] = int(final_left_bottom_y)
    final_right_top[1] = int(final_right_top_y)
    final_right_bottom[1] = int(final_right_bottom_y)

    # if -frame_width/2 < left_top_x < frame_width/2:
    final_left_top[0] = int(final_left_top_x)
    # if -frame_width/2 < left_bottom_x < frame_width/2:
    final_left_bottom[0] = int(final_left_bottom_x)

    # if frame_width/2 < right_top_x < 3/2 * frame_width:
    final_right_top[0] = int(final_right_top_x)   # + frame_width//2
    # if frame_width/2 < right_bottom_x < 3/2 * frame_width:
    final_right_bottom[0] = int(final_right_bottom_x)   # + frame_width//2
    # print("final_right_top:", [final_right_top_x, final_right_top_y])
    # print("final_right_bottom:", [final_right_bottom_x, final_right_bottom_y])

    # Drawing the final lines on the original frame
    copy_final_frame = small.copy()
    copy_final_frame = cv2.line(copy_final_frame, final_left_top, final_left_bottom, (50, 50, 250), 2)
    copy_final_frame = cv2.line(copy_final_frame, final_right_top, final_right_bottom, (50, 250, 50), 2)
    cv2.imshow('Final', copy_final_frame)

    # 12. Break the loop if 'q' is pressed or the video is finished
    # if cv2.waitKey(1) & 0xFF == ord('q'):
    #     break
    # end program
    if cv2.waitKey(10) & 0xFF == ord('q'):
        break

# Release the video capture object and close all windows
cam.release()
cv2.destroyAllWindows()
