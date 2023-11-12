print("Booting up…")

from used_model import MLP
import torch
import random as rng
from glob import glob
import cv2 as cv
import time
from math import trunc
import numpy
from used_model import CROPPED_WIDTH, CROPPED_HEIGHT, preprocess_input, CLASSES, load_snapshot

model = MLP(int(CROPPED_WIDTH/2 * CROPPED_HEIGHT/2), len(CLASSES))

print("Loading snapshot…")

generation, model, epochs, loss, accuracy = load_snapshot(model)

print(f"Generation: {generation}")
print(f"Epochs trained: {epochs}")
print(f"Validation loss: {loss:.3f}")
print(f"Validation accuracy: {accuracy*100:.2f} %")

# Set the model to evaluation mode
model.eval()

success = True

vidObj = cv.VideoCapture(rng.choice(glob('../*.mp4')))
# vidObj = cv.VideoCapture('../murgtalbahn.mp4')
# seek to random position
vidObj.set(cv.CAP_PROP_POS_FRAMES, rng.randint(0, vidObj.get(cv.CAP_PROP_FRAME_COUNT)))

# Used as counter variable
count = 0

# checks whether frames were extracted
success = 1

directions_of_last_second = [float('nan')] * int(vidObj.get(cv.CAP_PROP_FPS))

while success:
    # vidObj object calls read
    # function extract frames
    success, image = vidObj.read()
    count += 1
    preprocessed_input = preprocess_input(image)  # Preprocess the input as needed

    # Forward pass to obtain predicted outputs
    with torch.no_grad():
        # inputs = torch.tensor(preprocessed_input)  # Convert preprocessed input to a PyTorch tensor
        outputs = model(preprocessed_input)  # Get the predicted outputs from the model

    # Process the predicted outputs
    probabilities = torch.softmax(outputs, dim=1)

    # Get the predicted class labels
    _, predicted_label = torch.max(probabilities, dim=1)
    predicted_label = predicted_label.item()

    directions_of_last_second.pop(0)
    directions_of_last_second.append(predicted_label)

    # write predicted label onto image
    cv.putText(image, CLASSES[predicted_label], (10, 30), cv.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2, cv.LINE_AA)
    avg_direction = CLASSES[round(numpy.nanmean(directions_of_last_second))]
    cv.putText(image, f'smoothed: {avg_direction}', (10, 60), cv.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2, cv.LINE_AA)

    cv.imshow('img', image)
    pressed_key = cv.waitKey(1)

    if pressed_key == ord('q'):
        break
    elif pressed_key == ord('i'):
        # Convert the predicted labels to their respective class names
        predicted_class_names = {CLASSES[index]: value * 100 for index, value in enumerate(probabilities.tolist()[0])}
        print(f'Frame {count}: {predicted_class_names}')
    elif pressed_key == ord('s'):
        print(f'Saving frame {count}...')
        cv.imwrite(f'../labeled_images/milestones/frame-{count}-{trunc(time.time())}.jpg', image)
