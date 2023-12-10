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
import os
import albumentations as A

from used_model import CROPPED_WIDTH, CROPPED_HEIGHT, BATCH_SIZE, CLASSES, CLASS_LABEL_TO_INDEX, load_snapshot, MODEL_INPUT_WIDTH_HEIGHT

import torch.distributed as dist
import torch.multiprocessing as mp
import csv

def calculate_accuracy(y_pred, y):
    top_pred = y_pred.argmax(1, keepdim=True)
    correct = top_pred.eq(y.view_as(top_pred)).sum()
    acc = correct.float() / y.shape[0]
    return acc

def train(model, iterator, optimizer, criterion, device):
    epoch_loss = 0
    epoch_acc = 0

    model.train()

    for (x, y) in iterator:
        x = x.to(device)
        y = torch.as_tensor(y, device=device)

        optimizer.zero_grad()

        y_pred = model(x)

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
        for (x, y) in iterator:
            x = x.to(device)
            y = y.to(device)

            y_pred = model(x)

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

def save_snapshot(generation, model, epoch, loss, accuracy):
    snapshot = {
        "GENERATION": generation,
        "EPOCHS_RUN": epoch,
        "LOSS": loss,
        "TEST_ACCURACY": accuracy
    }

    write_csv_logfile(snapshot)

    snapshot["MODEL_STATE"] = model.state_dict()

    torch.save(snapshot, "saved-model.pt")
    print(f"Epoch: {epoch} | Training snapshot saved at saved-model.pt")


def write_csv_logfile(snapshot):
    is_new_file = not os.path.exists('training_log.csv')

    with open('training_log.csv', 'a') as f:
        w = csv.DictWriter(f, snapshot.keys())
        if is_new_file:
            w.writeheader()
        w.writerow(snapshot)

def main(rank, world_size):
    dist.init_process_group(backend="gloo", rank=rank, world_size=world_size)

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

    # distribute files in a 4/16 to 12/16 ratio (~ 25% test data)
    for filename, label in file_to_label.items():
        hex_hash = hashlib.sha256(str.encode(filename)).hexdigest()

        if hex_hash[0] == 'a' or hex_hash[0] == 'b' or hex_hash[0] == 'c' or hex_hash[0] == 'd':
            test_images.append((filename, label))
        else:
            training_images.append((filename, label))

    if rank == 0:
        print('Test images: ' + str(len(test_images)))
        print('Training images: ' + str(len(training_images)))

    class CustomDataset(torch.utils.data.Dataset):
        def __init__(self, data, is_test=False):
            self.data = data
            self.is_test = is_test
            self.transform = A.Compose([
                A.RandomBrightnessContrast(p=0.2),
                A.RandomShadow(p=0.2),
                A.RandomFog(p=0.2),
                A.RandomRain(p=0.2),
                A.Rotate(p=0.5, limit=4),
            ])

        def __len__(self):
            return len(self.data)

        def __getitem__(self, index):
            sample, target = self.data[index]

            from used_model import preprocess_input

            # load image
            img = cv.imread(sample)

            # augment input
            if not self.is_test:
                img = self.transform(image=img)["image"]

            tensor = preprocess_input(img)

            return tensor, CLASS_LABEL_TO_INDEX[target]

    training_loader = CustomDataset(training_images, False)
    test_loader = CustomDataset(test_images, True)

    train_sampler = torch.utils.data.distributed.DistributedSampler(training_loader)
    test_sampler = torch.utils.data.distributed.DistributedSampler(test_loader)

    train_iterator = torch.utils.data.DataLoader(training_loader,
                                                sampler=train_sampler,
                                                batch_size=BATCH_SIZE)

    test_iterator = torch.utils.data.DataLoader(test_loader,
                                                sampler=test_sampler,
                                                batch_size=BATCH_SIZE)

    from used_model import MLP

    model = MLP((MODEL_INPUT_WIDTH_HEIGHT ** 2), len(CLASSES))

    best_valid_loss = float('inf')
    best_accuracy = 0.0
    epochs = 0
    generation = 0

    if os.path.isfile('saved-model.pt'):
        generation, model, epochs, best_valid_loss, best_accuracy = load_snapshot(model)

    optimizer = torch.optim.Adam(model.parameters())

    criterion = torch.nn.CrossEntropyLoss()

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = model.to(device)
    criterion = criterion.to(device)

    EPOCHS_TO_TRAIN = 50

    for epoch in trange(EPOCHS_TO_TRAIN, disable=rank != 0):
        if rank == 0:
            print(f'\nEpoch: {epoch+1} ({epochs+epoch+1:02}) ', end='', flush=True)

        start_time = time.monotonic()

        train_sampler.set_epoch(epoch)
        test_sampler.set_epoch(epoch)

        train_loss, train_acc = train(model, train_iterator, optimizer, criterion, device)
        valid_loss, valid_acc = evaluate(model, test_iterator, criterion, device)
        end_time = time.monotonic()

        if rank == 0:
            epoch_mins, epoch_secs = epoch_time(start_time, end_time)

            print(f'| Epoch Time: {epoch_mins}m {epoch_secs}s')
            print(f'\tTrain Loss: {train_loss:.3f} | Train Acc: {train_acc*100:.2f}%')
            print(f'\t Val. Loss: {valid_loss:.3f} |  Val. Acc: {valid_acc*100:.2f}%')

            if valid_loss < best_valid_loss or valid_acc > best_accuracy:
                best_valid_loss = valid_loss
                best_accuracy = valid_acc
                save_snapshot(generation + 1, model, epochs+epoch+1, best_valid_loss, valid_acc)

if __name__ == "__main__":
    world_size = 8 # Define the number of processes
    mp.spawn(main, args=(world_size,), nprocs=world_size)