import cv2 as cv
from compositor.detectors.in_tunnel import InTunnelDetector

def inTunnelDetectorTest():
    print('\n-- TEST InTunnelDetector --\n')

    test_images = {
        'labeled_images/milestones/JPEGImages/frame-92-1702174287.jpg': False,
        'labeled_images/milestones/JPEGImages/frame-94-1702174287.jpg': False
    }

    detector = InTunnelDetector()
    detector.init(30)

    failures = []

    for image_path, expected_result in test_images.items():
        frame = cv.imread(image_path)
        features = detector.detect_features(frame)

        if features[0].value != expected_result:
            print('E', end='', flush=True)
            failures.append(image_path)
        else:
            print('.', end='', flush=True)

    print(f'\n\nFailures: {failures}')

    print('\n-- TEST END InTunnelDetector --\n')

if __name__ == '__main__':
    inTunnelDetectorTest()