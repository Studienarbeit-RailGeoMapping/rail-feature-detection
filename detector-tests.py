import cv2 as cv
from compositor.detectors.in_tunnel import InTunnelDetector
from time import time
import numpy as np

def timed_detector_run(detector, frame):
    REPS = 10000

    start = time()
    result = None
    for i in range(0, REPS):
        result = detector.detect_features(frame)
    end = time()
    return ((end - start) / REPS, result)

def inTunnelDetectorTest():
    print('\n-- TEST InTunnelDetector --\n')

    test_images = {
        'labeled_images/milestones/JPEGImages/frame-92-1702174287.jpg': False,
        'labeled_images/milestones/JPEGImages/frame-94-1702174287.jpg': False,
        'labeled_images/milestones/JPEGImages/frame-77123-1713956321.jpg': False,
        'labeled_images/milestones/JPEGImages/frame-77126-1713956321.jpg': False,
        'labeled_images/milestones/JPEGImages/frames1.jpg': False,
        'labeled_images/milestones/JPEGImages/frames2.jpg': False,
        'labeled_images/milestones/JPEGImages/frame-129598-1713958553.jpg': False,
        'labeled_images/milestones/JPEGImages/frame-129604-1713958553.jpg': False,
        'labeled_images/milestones/JPEGImages/frame-129614-1713958554.jpg': False,
        'labeled_images/milestones/JPEGImages/frame-129619-1713958554.jpg': False,
        'labeled_images/milestones/JPEGImages/frame-129626-1713958554.jpg': False,
        'labeled_images/milestones/JPEGImages/frame-129642-1713958555.jpg': False,
        'labeled_images/milestones/JPEGImages/frame-129646-1713958555.jpg': False,
        'labeled_images/milestones/JPEGImages/frame-129651-1713958555.jpg': False,
        'labeled_images/milestones/JPEGImages/frame-129676-1713958556.jpg': False,
    }

    detector = InTunnelDetector()
    detector.init(30)

    times = []
    failures = []

    for image_path, expected_result in test_images.items():
        frame = cv.imread(image_path)
        avg_time, result = timed_detector_run(detector, frame)
        times.append(avg_time)

        if result[0].value != expected_result:
            print('E', end='', flush=True)
            failures.append(image_path)
        else:
            print('.', end='', flush=True)

    print(f'\n\nFailures: {failures}')

    avg_iteration = np.mean(times)
    print(f'\n\nAverage times of all runs: {avg_iteration:.2} s')

    print('\n-- TEST END InTunnelDetector --\n')

if __name__ == '__main__':
    inTunnelDetectorTest()