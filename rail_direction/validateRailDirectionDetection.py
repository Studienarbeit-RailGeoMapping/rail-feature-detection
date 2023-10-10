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
    "failed_direction": 0,
    "failed_curve_strength": 0,
    "failed_none": 0,
}

test_failed_directions = []

test_start_time = time.time()

for path, expected_result in labels.items():
    # the implementation does not return the direction yet
    # TODO: The implementation should also return the direction

    result = get_rail_direction_from_path(path)

    if result == expected_result:
        test_results["passed"] += 1
        print('.', end='', flush=True)
    else:
        test_results["failed"] += 1

        if result is None:
            print(' ', end='', flush=True)
            test_results["failed_none"] += 1 # returning None (because of insecurity in the algorithm) is better than false results
        else:
            print('F', end='', flush=True)

            pos_of_underscore = result.find('_')
            if pos_of_underscore != -1:
                [curve_strength, direction] = result[:pos_of_underscore], result[pos_of_underscore+1:]

                pos_of_underscore = expected_result.find('_')
                if pos_of_underscore != -1:
                    [expected_curve_strength, expected_direction] = expected_result[:pos_of_underscore], expected_result[pos_of_underscore+1:]

                    if expected_direction != direction:
                        test_results["failed_direction"] += 1
                        test_failed_directions.append(path)

                    if expected_curve_strength != curve_strength:
                        test_results["failed_curve_strength"] += 1
                else:
                    test_results["failed_direction"] += 1
                    test_failed_directions.append(path)
            else:
                test_results["failed_direction"] += 1
                test_failed_directions.append(path)

test_duration = time.time() - test_start_time

print()
print(test_results)

print(f'\nTests took {test_duration:.3} s')

print(f'Failed images: {test_failed_directions}')