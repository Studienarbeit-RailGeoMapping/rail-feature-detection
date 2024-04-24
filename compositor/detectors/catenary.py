from compositor.models.features.text.text_feature import TextFeature
from .base import BaseDetector
import logging
import cv2 as cv
import numpy
import collections

logger = logging.getLogger(__name__)

TOP_AREA_WITH_CATENARY_WIDTH = 600
TOP_AREA_WITH_CATENARY_HEIGHT = 250
PADDING_TOP_AREA = 25

last_results = collections.deque(maxlen=100)

DEBUG = False

class CatenaryDetector(BaseDetector):
    def init(self, fps):
        super().init(fps)

    def detect_features(self, frame):
        _image_height, image_width, _color_channels = frame.shape

        # crop image to only include the center top part
        right = int((image_width - TOP_AREA_WITH_CATENARY_WIDTH) / 2)

        catenary_view = frame[PADDING_TOP_AREA:PADDING_TOP_AREA+TOP_AREA_WITH_CATENARY_HEIGHT, right:right+TOP_AREA_WITH_CATENARY_WIDTH]

        # increase contrast using lookup-table
        # catenary_view = cv.addWeighted(catenary_view, 1.5, catenary_view, 0, 0)

        # catenary_view = cv.medianBlur(catenary_view, 5)

        lower = (0,0,0)
        upper = (220, 220, 220)
        catenary_view_black_white = cv.inRange(catenary_view, lower, upper)

        if DEBUG:
            cv.imshow("catenary view black/white", catenary_view_black_white)
            cv.waitKey(1)

        # detect noise
        kernel = numpy.ones((3, 1), numpy.uint8) # vertical kernel to connect split lines
        noise = cv.dilate(catenary_view_black_white, kernel, iterations=1)
        noise = cv.erode(catenary_view_black_white, kernel, iterations=1)

        kernel_size = (7, 6) # should roughly have the size of the elements you want to remove
        kernel_el = cv.getStructuringElement(cv.MORPH_RECT, kernel_size)
        noise = cv.erode(noise, kernel_el, (-1, -1))
        noise = cv.dilate(noise, kernel_el, (-1, -1))

        if DEBUG:
            cv.imshow("detected noise", noise)
            cv.waitKey(1)

        percentage_of_noise = ((noise == 255).sum() / (TOP_AREA_WITH_CATENARY_WIDTH * TOP_AREA_WITH_CATENARY_HEIGHT))

        if percentage_of_noise > 0.95:
            return [TextFeature("Catenary detected", "too much noise")]

        # remove noise from image
        catenary_view_black_white[noise == 255] = 0

        if DEBUG:
            cv.imshow("catenary view thresholding", catenary_view_black_white)
            cv.waitKey(1)

        # Apply edge detection method on the image
        # edges = cv.Canny(cv.merge((catenary_view, catenary_view, catenary_view)), 50, 150, apertureSize=3)

        lines = cv.HoughLinesP(catenary_view_black_white, 1, numpy.pi/90, 1, 25, 25)

        detected_lines = 0

        if lines is not None:
            for line in lines:
                x1, y1, x2, y2 = line[0]

                if self.is_detected_line_possible_catenary((x1, y1), (x2, y2)):
                    detected_lines += 1
                    cv.line(catenary_view,(x1,y1),(x2,y2),(0,0,255),2)

        if DEBUG:
            cv.imshow("catenary view", catenary_view)
            cv.waitKey(1)

        result = detected_lines > 0

        last_results.append(result)

        return [TextFeature("Catenary detected", str(round(numpy.mean(last_results)) == 1))]

    def is_detected_line_possible_catenary(self, start, end):
        x_width = abs(start[0]-end[0])

        # completely straight line or too wide to be a catenary
        if x_width == 0 or x_width > 10:
            return False

        # too long (should detect GSM-R masts such as around frame 62700 in diesel video)
        # if abs(start[1]-end[1]) > 20:
        #     return False

        return True

# TODO: Write tests with example images