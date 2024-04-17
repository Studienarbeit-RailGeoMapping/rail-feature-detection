from ..label import Label
import cv2 as cv

class BooleanLabel(Label):
    def __init__(self, name, value=False) -> None:
        self.name = name
        self.value = value

    def draw_to_frame(self, frame, expected_y_start):
        name_repr = self.name + ': '

        width_of_label = cv.getTextSize(name_repr, cv.FONT_HERSHEY_SIMPLEX, 1, 2)[0][0]
        cv.putText(frame, name_repr, (10, expected_y_start), cv.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2, cv.LINE_AA)

        color_of_value = (0, 255, 0) if self.value else (0, 0, 255)
        cv.putText(frame, str(self.value), (10 + width_of_label, expected_y_start), cv.FONT_HERSHEY_SIMPLEX, 1, color_of_value, 2, cv.LINE_AA)

        return frame

    def to_dict(self):
        return {
            self.name: self.value
        }