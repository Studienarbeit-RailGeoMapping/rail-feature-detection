from abc import abstractmethod

class Label():
    @classmethod
    @abstractmethod
    def draw_to_frame(self, frame):
        return frame

    def to_dict(self):
        return {}