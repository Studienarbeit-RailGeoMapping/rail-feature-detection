from abc import abstractmethod

class Label():
    @classmethod
    @abstractmethod
    def draw_to_frame(frame):
        return frame