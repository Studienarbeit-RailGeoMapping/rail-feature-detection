from compositor.models.features.boolean.boolean_feature import BooleanFeature
from .base import BaseDetector
import logging
import cv2 as cv

logger = logging.getLogger(__name__)

TOP_AREA_WIDTH = 250
TOP_AREA_HEIGHT = 250
PADDING_TOP_AREA = 25

DEBUG = False
class InTunnelDetector(BaseDetector):
    def init(self, fps):
        super().init(fps)

    def detect_features(self, frame):
        _image_height, image_width, _color_channels = frame.shape

        # crop image to only include the center top part
        right = int((image_width - TOP_AREA_WIDTH) / 2)

        top_view = frame[PADDING_TOP_AREA:PADDING_TOP_AREA+TOP_AREA_HEIGHT, right:right+TOP_AREA_WIDTH]

        ret, black_and_white = cv.threshold(cv.cvtColor(
            top_view, cv.COLOR_BGR2GRAY), 100, 255, cv.THRESH_BINARY_INV)
        non_zero = cv.countNonZero(black_and_white)

        img_size = TOP_AREA_HEIGHT * TOP_AREA_WIDTH
        ratio_of_white_pixels = (img_size - non_zero) / img_size

        if DEBUG:
            print(ratio_of_white_pixels)
            cv.imshow('frame', frame)
            cv.imshow('in_tunnel', black_and_white)
            cv.waitKey(0)

        return [BooleanFeature("In Tunnel", ratio_of_white_pixels < 0.25)]

# TODO: Write tests with example images