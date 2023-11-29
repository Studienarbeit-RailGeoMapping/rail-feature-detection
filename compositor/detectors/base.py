from abc import abstractmethod
import logging

logger = logging.getLogger(__name__)

class BaseDetector():

    @classmethod
    @abstractmethod
    def init(self, fps):
        logger.info(f"Initializing {self.__name__}…")
        pass

    @classmethod
    @abstractmethod
    def detect_features(self):
        pass