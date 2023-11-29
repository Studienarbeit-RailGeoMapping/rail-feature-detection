from compositor.models.features.boolean.boolean_feature import BooleanFeature
from .base import BaseDetector
import logging
import cv2 as cv

logger = logging.getLogger(__name__)

class InTunnelDetector(BaseDetector):
    def init(self, fps):
        super().init(fps)


    def detect_features(self, frame):
        ret, black_and_white = cv.threshold(cv.cvtColor(
            frame, cv.COLOR_BGR2GRAY), 127, 255, cv.THRESH_BINARY_INV)
        non_zero = cv.countNonZero(black_and_white)

        img_size = frame.shape[0] * frame.shape[1]
        ratio_of_white_pixels = (img_size - non_zero) / img_size

        return [BooleanFeature("In Tunnel", ratio_of_white_pixels < 0.15)]