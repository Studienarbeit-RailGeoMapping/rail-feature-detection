from glob import glob
import cv2 as cv
import hashlib
import numpy
import sys
import time
import torch
import torchvision
import torchvision.transforms as transforms
from tqdm.notebook import trange, tqdm

from used_model import CROPPED_WIDTH, CROPPED_HEIGHT, BATCH_SIZE, CLASSES, CLASS_LABEL_TO_INDEX

file_to_label = {}
label_counts = {klass: 0 for klass in CLASSES}

with open('../labeled_images/directions/labelmap.txt') as fd:
    for line in fd.readlines():
        [file_name, label] = line.split(':')
        label = label.rstrip()

        file_to_label[file_name] = label
        label_counts[label] += 1

print(label_counts)

test_images = []
training_images = []

# distribute files in a 3/16 to 13/16 ratio (~ 18% test data)
for filename, label in file_to_label.items():
    hex_hash = hashlib.sha256(str.encode(filename)).hexdigest()

    if hex_hash[0] == 'a' or hex_hash[0] == 'b' or hex_hash[0] == 'c':
        test_images.append((filename, label))
    else:
        training_images.append((filename, label))

print('Test images: ' + str(len(test_images)))
print('Training images: ' + str(len(training_images)))

class CustomDataset(torch.utils.data.Dataset):
    def __init__(self, data, is_test=False):
        self.data = data
        self.is_test = is_test
    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        sample, target = self.data[index]

        from used_model import preprocess_input

        # load image
        img = cv.imread(sample)

        tensor = preprocess_input(img)

        return tensor, CLASS_LABEL_TO_INDEX[target]

training_loader = CustomDataset(training_images, False)
test_loader = CustomDataset(test_images, True)

train_iterator = torch.utils.data.DataLoader(training_loader,
                                 shuffle=True,
                                 batch_size=BATCH_SIZE)

test_iterator = torch.utils.data.DataLoader(test_loader,
                                batch_size=BATCH_SIZE)

def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

from used_model import MLP

model = MLP(CROPPED_WIDTH * CROPPED_HEIGHT, len(CLASSES))
model.load_state_dict(torch.load('saved-model.pt'))

optimizer = torch.optim.Adam(model.parameters())

criterion = torch.nn.CrossEntropyLoss()

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = model.to(device)
criterion = criterion.to(device)

def calculate_accuracy(y_pred, y):
    top_pred = y_pred.argmax(1, keepdim=True)
    correct = top_pred.eq(y.view_as(top_pred)).sum()
    acc = correct.float() / y.shape[0]
    return acc

def train(model, iterator, optimizer, criterion, device):
    epoch_loss = 0
    epoch_acc = 0

    model.train()

    for (x, y) in tqdm(iterator, desc="Training", leave=False):
        x = x.to(device)
        y = torch.as_tensor(y, device=device)

        optimizer.zero_grad()

        y_pred, _ = model(x)

        loss = criterion(y_pred, y)

        acc = calculate_accuracy(y_pred, y)

        loss.backward()

        optimizer.step()

        epoch_loss += loss.item()
        epoch_acc += acc.item()

    return epoch_loss / len(iterator), epoch_acc / len(iterator)


def evaluate(model, iterator, criterion, device):
    epoch_loss = 0
    epoch_acc = 0

    model.eval()

    with torch.no_grad():
        for (x, y) in tqdm(iterator, desc="Evaluating", leave=False):
            x = x.to(device)
            y = y.to(device)

            y_pred, _ = model(x)

            loss = criterion(y_pred, y)

            acc = calculate_accuracy(y_pred, y)

            epoch_loss += loss.item()
            epoch_acc += acc.item()

    return epoch_loss / len(iterator), epoch_acc / len(iterator)

def epoch_time(start_time, end_time):
    elapsed_time = end_time - start_time
    elapsed_mins = int(elapsed_time / 60)
    elapsed_secs = int(elapsed_time - (elapsed_mins * 60))
    return elapsed_mins, elapsed_secs

EPOCHS = 10

best_valid_loss = float('inf')

for epoch in trange(EPOCHS):
    start_time = time.monotonic()

    train_loss, train_acc = train(model, train_iterator, optimizer, criterion, device)
    valid_loss, valid_acc = evaluate(model, test_iterator, criterion, device)

    if valid_loss < best_valid_loss:
        best_valid_loss = valid_loss
        torch.save(model.state_dict(), 'saved-model.pt')

    end_time = time.monotonic()

    epoch_mins, epoch_secs = epoch_time(start_time, end_time)

    print(f'Epoch: {epoch+1:02} | Epoch Time: {epoch_mins}m {epoch_secs}s')
    print(f'\tTrain Loss: {train_loss:.3f} | Train Acc: {train_acc*100:.2f}%')
    print(f'\t Val. Loss: {valid_loss:.3f} |  Val. Acc: {valid_acc*100:.2f}%')