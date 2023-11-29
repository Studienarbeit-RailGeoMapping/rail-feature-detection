from compositor.models.features.text.text_label import TextLabel
from compositor.models.features.label import Label
from ..feature import Feature

class TextFeature(Feature):
    def __init__(self, name: str, value: str) -> None:
        self.name = name
        self.value = value

    def __repr__(self) -> str:
        return f"TextFeature('{self.name}', value='{self.value}')"

    def to_label(self) -> Label:
        return TextLabel(self.name, self.value)