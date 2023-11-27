import torch
import cv2 as cv
import numpy
import albumentations as A
import torchvision.transforms as transforms

CROPPED_WIDTH = 500
CROPPED_HEIGHT = 500
BATCH_SIZE = 64


CLASSES = ['sharp_left', 'slight_left', 'straight', 'slight_right', 'sharp_right']
# Define a dictionary to map class labels to numerical values
CLASS_LABEL_TO_INDEX = {label: index for index, label in enumerate(CLASSES)}

class MLP(torch.nn.Module):
    def __init__(self, input_dim, output_dim):
        super().__init__()

        self.input_fc = torch.nn.Linear(input_dim, 512)
        self.hidden_fc_1 = torch.nn.Linear(512, 256)
        self.drop_1 = torch.nn.Dropout(0.01)
        self.hidden_fc_2 = torch.nn.Linear(256, 100)
        self.drop_2 = torch.nn.Dropout(0.01)
        self.output_fc = torch.nn.Linear(100, output_dim)

    def forward(self, x):
        import torch.nn.functional as F
        # x = [batch size, height, width]

        batch_size = x.shape[0]

        x = x.view(batch_size, -1)

        # x = [batch size, height * width]

        h_1 = F.relu(self.input_fc(x))

        # h_1 = [batch size, 250]

        h_2 = self.drop_1(F.relu(self.hidden_fc_1(h_1)))

        h_3 = self.drop_2(F.relu(self.hidden_fc_2(h_2)))

        y_pred = self.output_fc(h_3)

        # y_pred = [batch size, output dim]

        return y_pred

def preprocess_input(img) -> torch.Tensor:
    image_height, image_width, _color_channels = img.shape

    # crop image to only include center
    right = int((image_width - CROPPED_WIDTH) / 2)
    top = int((image_height - CROPPED_HEIGHT) / 2)

    img = img[top:top+CROPPED_HEIGHT, right:right+CROPPED_WIDTH]

    img = cv.resize(img, (int(CROPPED_WIDTH / 2), int(CROPPED_HEIGHT / 2)), interpolation=cv.INTER_LINEAR)
    img = cv.cvtColor(img, cv.COLOR_BGR2GRAY)

    # img = cv.convertScaleAbs(img, 2, 2)

    img = cv.medianBlur(img, 3)

    # img = cv.adaptiveThreshold(
    #     img, 255, cv.ADAPTIVE_THRESH_GAUSSIAN_C, cv.THRESH_BINARY_INV, 7, 17)

    kernel = numpy.ones((3, 1), numpy.uint8) # vertical kernel to connect split lines
    img = cv.dilate(img, kernel, iterations=1)
    img = cv.erode(img, kernel, iterations=1)

    tensor = transforms.ToTensor()(img)
    return tensor


def load_snapshot(model):
    snapshot = torch.load("saved-model.pt")
    model.load_state_dict(snapshot["MODEL_STATE"])

    return snapshot["GENERATION"], model, snapshot["EPOCHS_RUN"], snapshot["LOSS"], snapshot["TEST_ACCURACY"]