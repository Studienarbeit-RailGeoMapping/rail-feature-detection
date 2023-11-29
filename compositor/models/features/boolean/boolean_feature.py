from compositor.models.features.boolean.boolean_label import BooleanLabel
from compositor.models.features.label import Label
from ..feature import Feature

class BooleanFeature(Feature):
    def __init__(self, name, value: bool=False) -> None:
        self.name = name
        self.value = value

    def __repr__(self) -> str:
        return f"BooleanFeature('{self.name}', value={self.value})"

    def to_label(self) -> Label:
        return BooleanLabel(self.name, self.value)