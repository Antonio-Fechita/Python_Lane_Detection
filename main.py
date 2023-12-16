import cv2
import numpy as np

cam = cv2.VideoCapture('Lane Detection Test Video 01.mp4')

left_top_x = left_bottom_x = right_top_x = right_bottom_x = left_top_y = left_bottom_y = right_top_y = right_bottom_y = 0

while True:

    ret, frame = cam.read()

    if ret is False:
        break

    frame = cv2.resize(frame, (480, 270))
    original_frame = frame.copy()

    # for row in range(0, len(frame)):
    #     for column in range(0, len(frame[0])):
    #         mean = (frame[row][column][0]/3 + frame[row][column][1]/3 + frame[row][column][2]/3)
    #         frame[row][column] = mean

    frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    height, width = frame.shape

    black_frame = np.zeros((height, width), dtype=np.uint8)

    upper_right = (width * 0.53, height * 0.77)
    upper_left = (width * 0.46, height * 0.77)
    lower_left = (0, height)
    lower_right = (width, height)

    corners = np.array([upper_right, upper_left, lower_left, lower_right], dtype=np.int32)
    cv2.fillConvexPoly(black_frame, corners, 1)

    street_frame = frame * black_frame
    cv2.imshow('Street view', street_frame)

    corners = np.float32(corners)
    frame_bounds = np.float32([(width, 0), (0, 0), (0, height), (width, height)])

    perspective_transform = cv2.getPerspectiveTransform(corners, frame_bounds)
    stretched_street = cv2.warpPerspective(street_frame, perspective_transform, (width, height))
    # cv2.imshow("Stretched street", stretched_street)
    blurred_stretched_street = cv2.blur(stretched_street, ksize=(5, 5))
    cv2.imshow("blurred", blurred_stretched_street)

    sobel_vertical = np.float32([[-1, -2, -1],
                                 [0, 0, 0],
                                 [+1, +2, +1]])
    sobel_horizontal = np.transpose(sobel_vertical)

    stretched_street = np.float32(stretched_street)
    img_horizontal_filter = cv2.filter2D(stretched_street, -1, sobel_horizontal)
    img_vertical_filter = cv2.filter2D(stretched_street, -1, sobel_vertical)

    # cv2.imshow("horizontal", cv2.convertScaleAbs(img_horizontal_filter))
    # cv2.imshow("vertical", cv2.convertScaleAbs(img_vertical_filter))

    final_edge_detection = np.sqrt(
        img_vertical_filter * img_vertical_filter + img_horizontal_filter * img_horizontal_filter)
    final_edge_detection = np.uint8(final_edge_detection)
    cv2.imshow("final edge detection", final_edge_detection)

    threshold = 60
    final_edge_detection[final_edge_detection >= threshold] = 255
    final_edge_detection[final_edge_detection < threshold] = 0

    cv2.imshow("absolute", final_edge_detection)

    removed_noise = final_edge_detection.copy()

    removed_noise[:, :int(width * 0.03)] = 0
    removed_noise[int(height * 0.95):, int(width // 3):int(width // 3 * 2)] = 0
    removed_noise[:, int(width * 0.97):] = 0

    cv2.imshow("removed noise", removed_noise)

    left_lane = removed_noise.copy()
    left_lane[:, int(width * 0.4):] = 0

    right_lane = removed_noise.copy()
    right_lane[:, :int(width * 0.4)] = 0

    # cv2.imshow("left", left_lane)
    # cv2.imshow("right", right_lane)

    left_ys, left_xs = np.transpose(np.argwhere(left_lane > 1))
    right_ys, right_xs = np.transpose(np.argwhere(right_lane > 1))


    if len(left_ys) > 0 and len(right_ys) > 0:
        left_line = np.polynomial.polynomial.polyfit(left_xs, left_ys, deg=1)
        right_line = np.polynomial.polynomial.polyfit(right_xs, right_ys, deg=1)

        temp_y = 0
        temp_x = (temp_y - left_line[0]) / left_line[1]
        if -(10 ** 8) < temp_x < 10 ** 8:
            left_top_x = temp_x
            left_top_y = temp_y

        temp_y = height
        temp_x = (temp_y - left_line[0]) / left_line[1]
        if -(10 ** 8) < temp_x < 10 ** 8:
            left_bottom_x = temp_x
            left_bottom_y = temp_y

        temp_y = 0
        temp_x = (temp_y - right_line[0]) / right_line[1]
        if -(10 ** 8) < temp_x < 10 ** 8:
            right_top_x = temp_x
            right_top_y = temp_y

        temp_y = height
        temp_x = (temp_y - right_line[0]) / right_line[1]
        if -(10 ** 8) < temp_x < 10 ** 8:
            right_bottom_x = temp_x
            right_bottom_y = temp_y

    left_top = int(left_top_x), int(left_top_y)
    left_bottom = int(left_bottom_x), int(left_bottom_y)
    right_top = int(right_top_x), int(right_top_y)
    right_bottom = int(right_bottom_x), int(right_bottom_y)

    cv2.line(removed_noise, left_top, left_bottom, (200, 0, 0), 5)
    cv2.line(removed_noise, right_top, right_bottom, (100, 0, 0), 5)

    cv2.imshow("lines", removed_noise)

    ###############

    empty_frame_left = np.zeros((height, width), dtype=np.uint8)
    cv2.line(empty_frame_left, left_top, left_bottom, (255, 0, 0), 3)
    perspective_transform = cv2.getPerspectiveTransform(frame_bounds, corners)
    empty_frame_left = cv2.warpPerspective(empty_frame_left, perspective_transform, (width, height))
    left_ys, left_xs = np.transpose(np.argwhere(empty_frame_left > 1))

    empty_frame_right = np.zeros((height, width), dtype=np.uint8)
    cv2.line(empty_frame_right, right_top, right_bottom, (255, 0, 0), 3)
    perspective_transform = cv2.getPerspectiveTransform(frame_bounds, corners)
    empty_frame_right = cv2.warpPerspective(empty_frame_right, perspective_transform, (width, height))
    right_ys, right_xs = np.transpose(np.argwhere(empty_frame_right > 1))

    for index in range(0, len(left_xs)):
        original_frame[left_ys[index], left_xs[index]] = (0, 0, 0)

    for index in range(0, len(right_xs)):
        original_frame[right_ys[index], right_xs[index]] = (50, 250, 50)

    cv2.imshow("final", original_frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cam.release()
cv2.destroyAllWindows()
