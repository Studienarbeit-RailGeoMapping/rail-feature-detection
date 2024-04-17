from .base import BaseDetector
from compositor.models.features.confidence.confidence_feature import ConfidenceFeature
from compositor.models.features.text.text_feature import TextFeature
from rail_direction_using_image_classification.used_model import preprocess_input, CLASSES
import collections
import logging
import numpy
import torch

logger = logging.getLogger(__name__)

last_results = collections.deque(maxlen=100)

class RailDirectionDetector(BaseDetector):
    def init(self, fps):
        super().init(fps)

        from rail_direction_using_image_classification.used_model import MLP, MODEL_INPUT_WIDTH_HEIGHT, load_snapshot

        model = MLP(MODEL_INPUT_WIDTH_HEIGHT ** 2, len(CLASSES))
        generation, model, epochs, loss, accuracy = load_snapshot(
            model,
            "rail_direction_using_image_classification/saved-model.pt"
        )

        self.model = model

        logger.info(
            f"Stats of used model: GEN {generation}, EPOCHS {epochs}, VAL. LOSS: {loss:.3f}, VAL. ACCURACY: {accuracy*100:.2f} %"
        )

    def detect_features(self, frame):
        preprocessed_input = preprocess_input(frame)

        # Forward pass to obtain predicted outputs
        with torch.no_grad():
            # inputs = torch.tensor(preprocessed_input)  # Convert preprocessed input to a PyTorch tensor
            outputs = self.model(preprocessed_input)  # Get the predicted outputs from the model

        # Process the predicted outputs
        probabilities = torch.softmax(outputs, dim=1)

        # Get the predicted class labels
        probability, predicted_label = torch.max(probabilities, dim=1)
        predicted_label = predicted_label.item()

        last_results.append(predicted_label)

        avg_direction = CLASSES[round(numpy.mean(last_results))]

        return [
            TextFeature("Rail direction", avg_direction),
            ConfidenceFeature("Rail direction", probability)
        ]
