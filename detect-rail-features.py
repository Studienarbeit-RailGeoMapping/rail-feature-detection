if __name__ == "__main__":
    print("Booting compositor…")

    import logging
    logging.basicConfig(level=logging.DEBUG, format='%(asctime)s %(levelname)s %(name)s: %(message)s')

    from concurrent.futures import ProcessPoolExecutor
    from math import trunc
    import time
    from compositor.detectors.in_tunnel import InTunnelDetector
    from compositor.detectors.rail_direction import RailDirectionDetector
    from compositor.detectors.catenary import CatenaryDetector
    import cv2 as cv
    from glob import glob
    import random as rng
    import concurrent


    # load detectors
    detectors = [
        # CatenaryDetector(),
        InTunnelDetector(),
        # RailDirectionDetector(),
    ]

    video_file_path = rng.choice(glob('./*.mp4'))
    video_file_path = 'führerstandsmitfahrt-diesel-freudenstadt-hausach.mp4'
    # video_file_path = './schwarzwaldbahn-karlsruhe-konstanz.mp4'
    video_file_path = './murgtalbahn.mp4'

    vidObj = cv.VideoCapture(video_file_path)

    frame_pos = rng.randint(0, vidObj.get(cv.CAP_PROP_FRAME_COUNT))
    frame_pos = 127700

    # seek to random position
    vidObj.set(cv.CAP_PROP_POS_FRAMES, frame_pos)

    fps = vidObj.get(cv.CAP_PROP_FPS)

    for detector in detectors:
        detector.init(fps=fps)

    logging.info(f"Playing {video_file_path} at {frame_pos} ({fps} fps)…")

    last_state = None

    RUN_SILENTLY = False
    FIVE_SECONDS_DURATION = 5*int(fps)

    with ProcessPoolExecutor() as executor:
        frame_count = frame_pos
        while True:
            success, frame = vidObj.read()
            frame_count += 1

            if not success:
                break

            futures = [executor.submit(detector.detect_features, frame) for detector in detectors]
            labels = []

            for future in concurrent.futures.as_completed(futures):
                try:
                    features = future.result()
                except Exception as exc:
                    logging.warning('%r generated an exception: %s' % (exc))
                else:
                    labels.extend(map(lambda x: x.to_label(), features))

            y_start = 30

            image = frame.copy()

            detected_features = {}

            for label in labels:
                if not RUN_SILENTLY:
                    image = label.draw_to_frame(image, expected_y_start=y_start)
                    y_start += 30

                detected_features.update(label.to_dict())

            current_state = {
                "frame": frame_count,
                "detected_features": detected_features
            }

            if last_state is None or last_state["detected_features"] != detected_features:
                last_state = current_state
                logging.info(last_state)


            if RUN_SILENTLY:
                if frame_count % FIVE_SECONDS_DURATION == 0:
                    logging.info(f'currently at frame {frame_count}')
            else:
                # show the frame
                cv.imshow('frame', image)
                pressed_key = cv.waitKey(1)

                if pressed_key == ord('q'):
                    break
                elif pressed_key == ord('s'):
                    logging.debug(f'Saving frame {frame_count}...')
                    cv.imwrite(f'labeled_images/milestones/frame-{frame_count}-{trunc(time.time())}.jpg', frame)

