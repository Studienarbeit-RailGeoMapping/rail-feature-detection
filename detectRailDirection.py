import cv2 as cv
import math
from glob import glob
import random as rng
import numpy

img_path = rng.choice(glob('labeled_images/milestones/JPEGImages/*.jpg'))

print(f'showing {img_path}')

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
    img, 255, cv.ADAPTIVE_THRESH_GAUSSIAN_C, cv.THRESH_BINARY_INV, 5, 15)

# ret, img = cv.threshold(img, 150, 255, cv.THRESH_BINARY)

cv.imshow('img', img)
cv.waitKey(0)


# # canny_output = cv.Canny(cv.cvtColor(
# #     img, cv.COLOR_BGR2GRAY), threshold, threshold * 2)

contours, hierarchy = cv.findContours(
    img, cv.RETR_LIST, cv.CHAIN_APPROX_SIMPLE)

# read colored image to draw contures on
contured_image = cv.imread(img_path)
contured_image = contured_image[top:top+cropped_height, right:right+cropped_width]

def std_to_category(std: int):
    if std > 4:
        return 'sharp'

    if std > 2:
        return 'slight'

    return 'straight'

angles = []

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
        return f"LineBoundingRect(y_height={self.y_height})"

lines = []

for i in range(len(contours)):
    x_start, y_start, x_width, y_height = cv.boundingRect(contours[i])

    # get all contours that touch the lower 5 %
    if y_start + y_height > cropped_height * 0.9:
        line = LineBoundingRect(x_start, y_start, x_width, y_height, contours[i], i)
        # print(line.get_area())

        lines.append(line)

if len(lines) > 2:
    # get rail by getting two longest lines
    rail = sorted(lines, key=lambda x: x.y_height, reverse=True)[0:2]

    standard_deviations = []
    i = 0

    for track in rail:
        random_color = (rng.randrange(0, 255), rng.randrange(0, 255), rng.randrange(0, 255))
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

        if len(change_rates) > 0:
            print(change_rates)
            std = numpy.std(change_rates)
            print(std)

            standard_deviations.append(std)

        i += 1

    mean_std_deviation = numpy.mean(standard_deviations)

    cv.putText(contured_image, std_to_category(mean_std_deviation), (10, 40), cv.FONT_HERSHEY_PLAIN, 2, (255, 0, 0), 3)



cv.imshow('img', contured_image)

cv.waitKey(0)