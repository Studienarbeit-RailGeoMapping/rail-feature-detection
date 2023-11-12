from glob import glob
import cv2 as cv
import logging
import math
import numpy
import random as rng
import sys
import os

def final_rail_direction_from_directions(directions: list) -> str|None:
    [first, second] = directions

    if first == second:
        return first

    return None

def get_rail_direction_from_path(img_path: str) -> str|None:
    logging.debug(f'showing {img_path}')

    img = cv.imread(img_path)

    image_height, image_width, _color_channels = img.shape

    # crop image to only include center
    cropped_width = 500
    cropped_height = 500

    right = int((image_width - cropped_width) / 2)
    top = int((image_height - cropped_height) / 2)

    img = img[top:top+cropped_height, right:right+cropped_width]
    img = cv.resize(img, (int(cropped_width / 2), int(cropped_height / 2)), interpolation=cv.INTER_LINEAR)
    img = cv.cvtColor(img, cv.COLOR_BGR2GRAY)

    img = cv.convertScaleAbs(img, 2, 2)

    img = cv.medianBlur(img, 3)

    # img = cv.adaptiveThreshold(
    #     img, 255, cv.ADAPTIVE_THRESH_GAUSSIAN_C, cv.THRESH_BINARY_INV, 7, 17)

    kernel = numpy.ones((3, 1), numpy.uint8) # vertical kernel
    img = cv.dilate(img, kernel, iterations=1)
    img = cv.erode(img, kernel, iterations=1)

    if logging.getLogger().isEnabledFor(logging.DEBUG):
        cv.imshow('img', img)
        cv.waitKey(0)


    # # canny_output = cv.Canny(cv.cvtColor(
    # #     img, cv.COLOR_BGR2GRAY), threshold, threshold * 2)

    contours, _hierarchy = cv.findContours(
        img, cv.RETR_LIST, cv.CHAIN_APPROX_SIMPLE)

    # read colored image to draw contures on
    colored_image = cv.imread(img_path)[top:top+cropped_height, right:right+cropped_width]

    contured_image = colored_image.copy()

    cv.imshow('img', contured_image)


    def std_to_category(std):
        tolerance = 0.2

        if std > 4 + tolerance:
            return 'sharp'

        if std > 1 + tolerance:
            return 'slight'

        return 'straight'

    class LineBoundingRect:
        def __init__(self, x_start, y_start, x_width, y_height, contour, contour_idx) -> None:
            self.x_start = x_start
            self.y_start = y_start
            self.x_width = x_width
            self.y_height = y_height
            self.contour = contour
            self.contour_idx = contour_idx

        def get_angle(self) -> float:
            return math.atan(self.y_height/self.x_width) * (180/math.pi)

        def get_area(self) -> float:
            return self.x_width * self.y_height

        def get_hypotenuse(self) -> float:
            return math.sqrt(x_width * x_width + y_height * y_height)

        def __repr__(self) -> str:
            return f"LineBoundingRect(y_height={self.y_height}, x_start={self.x_start})"

    lines = []

    for key in range(len(contours)):
        x_start, y_start, x_width, y_height = cv.boundingRect(contours[key])

        # get all contours that touch the lower 10 % and are not to far away from center
        if y_start + y_height > cropped_height * 0.9 and x_start + x_width < cropped_width * 0.9:
            line = LineBoundingRect(x_start, y_start, x_width, y_height, contours[key], key)
            # logging.debug(line.get_area())

            lines.append(line)

    if len(lines) <= 2:
        return None

    # get rail by getting two longest lines
    rail = sorted(lines, key=lambda x: x.y_height, reverse=True)[0:2]

    standard_deviations = []
    key = 0

    left_track = None
    right_track = None

    for track in rail:
        if left_track is None or left_track.x_start > track.x_start:
            left_track = track

        if right_track is None or right_track.x_start < track.x_start:
            right_track = track

    if left_track is None or right_track is None:
        return None

    if left_track == right_track:
        return None

    rail_direction = None
    track_with_more_contours = max([left_track, right_track], key=lambda x: len(x.contour))

    track_specific_x_to_y = {
        'left': {},
        'right': {}
    }

    for track in [left_track, right_track]:
        key = 'left' if track == left_track else 'right'

        random_color = (rng.randrange(0, 255), rng.randrange(0, 255), rng.randrange(0, 255))

        if logging.getLogger().isEnabledFor(logging.DEBUG):
            cv.drawContours(contured_image, track.contour, -1, random_color, 2, cv.LINE_4)

        for vec in track.contour:
            for x, y in vec:
                if x not in track_specific_x_to_y:
                    track_specific_x_to_y[key][x] = []

                track_specific_x_to_y[key][x].append(y)

        # calculate mean of all coordinates
        for x in track_specific_x_to_y[key]:
            track_specific_x_to_y[key][x] = numpy.mean(track_specific_x_to_y[key][x])

        change_rates = []

        min_y = min(track_specific_x_to_y[key].keys())
        max_y = max(track_specific_x_to_y[key].keys())

        for j in range(min_y, max_y):
            if j not in track_specific_x_to_y[key]:
                continue

            if j-1 not in track_specific_x_to_y[key]:
                continue

            change_rate = (track_specific_x_to_y[key][j - 1] - track_specific_x_to_y[key][j])

            change_rates.append(change_rate)

        if track == track_with_more_contours:
            y_diff_of_track = track_specific_x_to_y[key][max_y] - track_specific_x_to_y[key][min_y]

            logging.debug(f'y diff of track with more contours: {y_diff_of_track}')

            if y_diff_of_track < 0:
                rail_direction = 'right'
            else:
                rail_direction = 'left'

        if len(change_rates) > 1:
            logging.debug("change rates " + str(change_rates))
            std = numpy.std(change_rates, ddof=1)
            logging.debug("std of change rate " + str(std))

            standard_deviations.append(std)

    horizontal_space_between_rails = right_track.x_start - left_track.x_start
    logging.info(f"horizontal space between rails: {horizontal_space_between_rails} px")

    if horizontal_space_between_rails < 20 or horizontal_space_between_rails > 150:
        logging.info(f"too less/much difference between two rails -> no indication possible: {horizontal_space_between_rails} px")
        return None

    logging.debug(track_specific_x_to_y)

    # build trapezoid by finding areas where there are pixels on one height
    min_y = None
    max_y = None

    for x0, y0 in track_specific_x_to_y['left'].items():
        # find x1 with the same y as y0 on the other track
        x1 = next((x for x, y in track_specific_x_to_y['right'].items() if y == y0), None)
        if x1 is None:
            continue

        if min_y is None:
            min_y = (y0, x0, x1)
        elif min_y[0] > y0:
            min_y = (y0, x0, x1)

        if max_y is None:
            max_y = (y0, x0, x1)
        elif max_y[0] < y0:
            max_y = (y0, x0, x1)

    if min_y is None or max_y is None:
        logging.info("couldn't find parallel part for trapezoid")
        return None


    # draw trapezoid
    if logging.getLogger().isEnabledFor(logging.DEBUG):
        pts = numpy.int32([[min_y[1], min_y[0]], [max_y[1], max_y[0]], [max_y[2], max_y[0]], [min_y[2], min_y[0]]])
        pts = pts.reshape((-1,1,2))
        cv.polylines(contured_image, [pts], True, (255, 255), 3)

        rail_ties_view_height_width = 200
        rail_ties_view_padding = 0
        rail_ties_view = colored_image[int(min_y[0]):int(max_y[0]), max_y[1]:max_y[2]] # [top:top+cropped_height, right:right+cropped_width]

        if rail_ties_view.shape[0] == 0 or rail_ties_view.shape[1] == 0:
            print("trapezoid of height 0")
            return None

        cv.imshow('rail ties', rail_ties_view)
        cv.resizeWindow('rail ties', rail_ties_view_height_width * 3, rail_ties_view_height_width * 3)
        cv.moveWindow('rail ties', 400, 100)

        warp_pts = numpy.float32([
            [min_y[1] - rail_ties_view_padding, min_y[0]],
            [max_y[1] - rail_ties_view_padding, max_y[0]],
            [min_y[2] + rail_ties_view_padding, min_y[0]],
            [max_y[2] + rail_ties_view_padding, max_y[0]]
        ])
        target_pts = numpy.float32([[0,0], [0, rail_ties_view_height_width], [rail_ties_view_height_width, 0], [rail_ties_view_height_width, rail_ties_view_height_width]])
        M = cv.getPerspectiveTransform(warp_pts, target_pts)
        dst = cv.warpPerspective(colored_image, M, (rail_ties_view_height_width,rail_ties_view_height_width))
        cv.imshow('dst', dst)
        cv.moveWindow('dst', 600, 100)

    if len(standard_deviations) == 0:
        return None

    if len(standard_deviations) == 2 and abs(standard_deviations[1] - standard_deviations[0]) > 3:
        return None

    mean_std_deviation = numpy.mean(standard_deviations)
    logging.debug(f'mean std deviation: {mean_std_deviation}')

    logging.debug(f'left: {left_track}')
    logging.debug(f'right: {right_track}')
    logging.debug(f'detected directions: {rail_direction}')

    label = std_to_category(mean_std_deviation)

    if label != 'straight':
        label += f'_{rail_direction}'

    if logging.getLogger().isEnabledFor(logging.DEBUG):
        cv.putText(contured_image, label, (10, 40), cv.FONT_HERSHEY_PLAIN, 2, (255, 0, 0), 3)

    return label


if __name__ == '__main__':
    logging.basicConfig(level=logging.DEBUG)
    img_path = rng.choice(glob('../labeled_images/milestones/*.jpg'))

    while True:
        # img_path = '../labeled_images/milestones/JPEGImages/1692817253.jpg'
        result = get_rail_direction_from_path(img_path)

        pressed_key = cv.waitKey(0)
        print(pressed_key)

        label = None
        # slight right (Right)
        if pressed_key == 3:
            label = 'slight_right'
        # slight left (Left)
        elif pressed_key == 2:
            label = 'slight_left'
        # straight (Up)
        elif pressed_key == 0:
            label = 'straight'
        # sharp left (A)
        elif pressed_key == 97:
            label = 'sharp_left'
        # sharp left (S)
        elif pressed_key == 115:
            label = 'sharp_right'
        elif pressed_key == ord('q'):
            sys.exit(0)
        elif pressed_key == ord('r'):
            img_path = rng.choice(glob('../labeled_images/milestones/*.jpg'))
        else:
            logging.info(f'unknown key pressed: keycode={pressed_key}')

        if label is not None:
            with open("../labeled_images/directions/labelmap.txt", "a+") as myfile:
                new_image_path = f'../labeled_images/milestones/JPEGImages/{os.path.basename(img_path)}'
                os.rename(img_path, new_image_path)
                myfile.write(f"{new_image_path}:{label}\n")
                logging.info(f'\nlabeled as {label}…\n')
                img_path = rng.choice(glob('../labeled_images/milestones/*.jpg'))
