from compositor.models.features.text.text_label import TextLabel
from compositor.models.features.label import Label
from ..feature import Feature

class ConfidenceFeature(Feature):
    def __init__(self, name: str, value: float) -> None:
        self.name = name
        self.value = value

    def __repr__(self) -> str:
        return f"ConfidenceFeature('{self.name}', value='{self.value}')"

    def confidence_to_human_text(self):
        if self.value < 0.5:
            return "low"
        elif self.value < 0.75:
            return "medium"
        else:
            return "high"

    def to_label(self) -> Label:
        return TextLabel('Confidence ' + self.name, self.confidence_to_human_text())