from used_model import MLP
import torch
import random as rng
from glob import glob
import cv2 as cv

from used_model import CROPPED_WIDTH, CROPPED_HEIGHT, BATCH_SIZE, preprocess_input, CLASSES, CLASS_LABEL_TO_INDEX


model = MLP(CROPPED_WIDTH * CROPPED_HEIGHT, len(CLASSES))

model.load_state_dict(torch.load('saved-model.pt'))  # Load the trained model parameters

# Set the model to evaluation mode
model.eval()

success = True

vidObj = cv.VideoCapture('../full-video-1080.mp4')

# Used as counter variable
count = 0

# checks whether frames were extracted
success = 1

while success:
    # vidObj object calls read
    # function extract frames
    success, image = vidObj.read()
    count += 1
    preprocessed_input = preprocess_input(image)  # Preprocess the input as needed

    # Forward pass to obtain predicted outputs
    with torch.no_grad():
        # inputs = torch.tensor(preprocessed_input)  # Convert preprocessed input to a PyTorch tensor
        outputs, _ = model(preprocessed_input)  # Get the predicted outputs from the model

    # Process the predicted outputs
    probabilities = torch.softmax(outputs, dim=1)

    # Convert the predicted labels to their respective class names
    predicted_class_names = {CLASSES[index]: value * 100 for index, value in enumerate(probabilities.tolist()[0])}

    # Get the predicted class labels
    _, predicted_labels = torch.max(probabilities, dim=1)

    # write predicted label onto image
    cv.putText(image, CLASSES[predicted_labels.item()], (10, 30), cv.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2, cv.LINE_AA)

    cv.imshow('img', image)
    pressed_key = cv.waitKey(1)

    if pressed_key == ord('q'):
        break
    elif pressed_key == ord('i'):
        print(f'Frame {count}: {predicted_class_names}')
