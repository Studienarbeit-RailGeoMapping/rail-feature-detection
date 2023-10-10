from glob import glob
import cv2 as cv
import logging
import math
import numpy
import random as rng
import sys

def final_rail_direction_from_directions(directions: list) -> str|None:
    [first, second] = directions

    if first == second:
        return first

    return None

def get_rail_direction_from_path(img_path: str) -> str|None:
    logging.debug(f'showing {img_path}')

    img = cv.imread(img_path, cv.IMREAD_GRAYSCALE)

    image_height, image_width = img.shape

    # crop image to only include center
    cropped_width = 275
    cropped_height = 300

    right = int((image_width - cropped_width) / 2)
    top = int((image_height - cropped_height) / 2)

    img = img[top:top+cropped_height, right:right+cropped_width]

    img = cv.medianBlur(img, 3)

    img = cv.adaptiveThreshold(
        img, 255, cv.ADAPTIVE_THRESH_GAUSSIAN_C, cv.THRESH_BINARY_INV, 7, 17)

    if logging.getLogger().isEnabledFor(logging.DEBUG):
        cv.imshow('img', img)
        cv.waitKey(0)


    # # canny_output = cv.Canny(cv.cvtColor(
    # #     img, cv.COLOR_BGR2GRAY), threshold, threshold * 2)

    contours, _hierarchy = cv.findContours(
        img, cv.RETR_LIST, cv.CHAIN_APPROX_SIMPLE)

    # read colored image to draw contures on
    contured_image = cv.imread(img_path)
    contured_image = contured_image[top:top+cropped_height, right:right+cropped_width]

    def std_to_category(std: int):
        if std > 4:
            return 'sharp'

        if std > 1:
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

    for i in range(len(contours)):
        x_start, y_start, x_width, y_height = cv.boundingRect(contours[i])

        # get all contours that touch the lower 5 % and are not to far away from center
        if y_start + y_height > cropped_height * 0.9:
            line = LineBoundingRect(x_start, y_start, x_width, y_height, contours[i], i)
            # logging.debug(line.get_area())

            lines.append(line)

    if len(lines) <= 2:
        return None

    # get rail by getting two longest lines
    rail = sorted(lines, key=lambda x: x.y_height, reverse=True)[0:2]

    standard_deviations = []
    i = 0

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

    for track in [left_track, right_track]:
        if track == left_track:
            logging.debug('left')
        else:
            logging.debug('right')

        random_color = (rng.randrange(0, 255), rng.randrange(0, 255), rng.randrange(0, 255))

        if logging.getLogger().isEnabledFor(logging.DEBUG):
            cv.drawContours(contured_image, track.contour, -1, random_color, 2, cv.LINE_4)

        x_to_y = {}

        for vec in track.contour:
            for x, y in vec:
                if x not in x_to_y:
                    x_to_y[x] = []

                x_to_y[x].append(y)

        # calculate mean of all coordinates
        for x in x_to_y:
            x_to_y[x] = numpy.mean(x_to_y[x])

        change_rates = []

        min_x = min(x_to_y.keys())
        max_x = max(x_to_y.keys())

        for i in range(min_x, max_x):
            if i not in x_to_y:
                continue

            if i-1 not in x_to_y:
                continue

            change_rate = (x_to_y[i - 1] - x_to_y[i])

            change_rates.append(change_rate)

        if track == track_with_more_contours:
            if x_to_y[max_x] - x_to_y[min_x] < 0:
                rail_direction = 'right'
            else:
                rail_direction = 'left'

        if len(change_rates) > 1:
            logging.debug(change_rates)
            std = numpy.std(change_rates, ddof=1)
            logging.debug(std)

            standard_deviations.append(std)

        i += 1

    if len(standard_deviations) == 0:
        return None

    if len(standard_deviations) == 2 and abs(standard_deviations[1] - standard_deviations[0]) > 3:
        return None

    mean_std_deviation = numpy.mean(standard_deviations)
    logging.debug(f'mean std deviation: {mean_std_deviation}')

    horizontal_space_between_rails = right_track.x_start - left_track.x_start
    logging.info(f"horizontal space between rails: {horizontal_space_between_rails} px")

    # if horizontal_space_between_rails < 30 or horizontal_space_between_rails > 175:
    #     logging.info(f"too less/much difference between two rails -> no indication possible: {horizontal_space_between_rails} px")
    #     return None

    logging.debug(f'left: {left_track}')
    logging.debug(f'right: {right_track}')
    logging.debug(f'detected directions: {rail_direction}')

    label = std_to_category(mean_std_deviation)

    if label != 'straight':
        label += f'_{rail_direction}'

    if logging.getLogger().isEnabledFor(logging.DEBUG):
        cv.putText(contured_image, label, (10, 40), cv.FONT_HERSHEY_PLAIN, 2, (255, 0, 0), 3)
        cv.imshow('img', contured_image)

    return label


if __name__ == '__main__':
    logging.basicConfig(level=logging.DEBUG)

    while True:
        img_path = rng.choice(glob('../labeled_images/milestones/JPEGImages/*.jpg'))
        # img_path = '../labeled_images/milestones/JPEGImages/1692968178-1.jpg'
        result = get_rail_direction_from_path(img_path)

        pressed_key = cv.waitKey(0)

        label = None
        # slight right (Right)
        if pressed_key == 2:
            label = 'slight_right'
        # slight left (Left)
        elif pressed_key == 3:
            label = 'slight_left'
        # straight (Up)
        elif pressed_key == 0:
            label = 'straight'
        # sharp left (A)
        elif pressed_key == 97:
            label = 'sharp_left'
        # sharp left (B)
        elif pressed_key == 115:
            label = 'sharp_right'
        elif pressed_key == ord('q'):
            sys.exit(0)
        else:
            logging.info(f'unknown key pressed: keycode={pressed_key}')

        if result is not None and label is not None:
            with open("../labeled_images/directions/labelmap.txt", "a+") as myfile:
                myfile.write(f"{img_path}:{label}\n")
                logging.info(f'labeled as {label}…\n')

