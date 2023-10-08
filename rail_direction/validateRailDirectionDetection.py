from detectRailDirection import get_rail_direction_from_path
import time

labels = {}

with open("../labeled_images/directions/labelmap.txt", "r") as fd:
    for line in fd.readlines():
        path, label = line.split(':')
        path = path.strip()
        label = label.strip()

        labels[path] = label

test_results = {
    "passed": 0,
    "failed": 0,
}

test_failed_paths = []

test_start_time = time.time()

for path, expected_result in labels.items():
    # the implementation does not return the direction yet
    # TODO: The implementation should also return the direction
    if expected_result.startswith('slight_'):
        expected_result = 'slight'
    elif expected_result.startswith('sharp_'):
        expected_result = 'sharp'

    result = get_rail_direction_from_path(path)

    if result == expected_result:
        test_results["passed"] += 1
        print('.', end='', flush=True)
    else:
        test_results["failed"] += 1
        test_failed_paths.append(path)
        print('F', end='', flush=True)

test_duration = time.time() - test_start_time

print()
print(test_results)

print(f'\nTests took {test_duration:.3} s')

print(f'Failed images: {test_failed_paths}')