import os

import numpy as np

from duckiebots_unreal_sim.tools import UEHTTPBridge, ImageRenderer
from level_randomization import randomize_level

if __name__ == '__main__':
    RUNname = '0320'
    bridge = UEHTTPBridge()
    # process_manger = UEProcessManager(game_launcher_file_path=DEFAULT_GAME_LAUNCHER_PATH)
    # process_manger.launch_game(render_off_screen=False)

    image_renderer = ImageRenderer(height=84, width=84)  # default 256 x 256
    image_renderer2 = ImageRenderer(height=84, width=84)

    valid_location_xyz_low = (-2100.0, -1600.0, 30.0)
    valid_location_xyz_high = (2100.0, 1600.0, 200.0)

    valid_roll_pitch_yaw_low = (-10.0, -64.0, -180.0)
    valid_roll_pitch_yaw_high = (10, 30.0, 180.0)

    if not os.path.exists('data/frame'):
        os.makedirs('data/frame')
    if not os.path.exists('data/segment'):
        os.makedirs('data/segment')

    try:
        i = 0
        while True:
            if i % 10 == 0:
                randomize_level(http_bridge=bridge)
                print(f'count: {i}')

            location_xyz = tuple(np.random.uniform(low=valid_location_xyz_low, high=valid_location_xyz_high))
            roll_pitch_yaw = tuple(np.random.uniform(low=valid_roll_pitch_yaw_low, high=valid_roll_pitch_yaw_high))

            rgb_image, mask_image = bridge.capture_scene_at_location(
                location_xyz=location_xyz,
                roll_pitch_yaw=roll_pitch_yaw
            )
            if rgb_image is not None:
                # (Save rgb_image and mask_image here)

                np.save(f'data/frame/{RUNname}f{i}.npy', rgb_image)
                np.save(f'data/segment/{RUNname}f{i}.npy', mask_image)

                # show images
                image_renderer.render_cv2_image(cv2_image=rgb_image, channel_order="RGB")
                image_renderer2.render_cv2_image(cv2_image=mask_image, channel_order="RGB")
                i += 1

            if i == 10001:
                break

    except KeyboardInterrupt:
        print("stopping...")
    finally:
        pass
        image_renderer.close()
        image_renderer2.close()
        # process_manger.close()
    print("done")
