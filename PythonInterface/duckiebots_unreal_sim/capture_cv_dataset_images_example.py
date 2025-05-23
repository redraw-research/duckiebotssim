import numpy as np

from duckiebots_unreal_sim.level_randomization import randomize_level
from duckiebots_unreal_sim.tools import UEHTTPBridge, UEProcessManager, ImageRenderer, DEFAULT_GAME_LAUNCHER_PATH

if __name__ == '__main__':
    bridge = UEHTTPBridge()
    process_manger = UEProcessManager(game_launcher_file_path=DEFAULT_GAME_LAUNCHER_PATH)
    process_manger.launch_game(render_off_screen=False)

    image_renderer = ImageRenderer(height=256, width=256)
    image_renderer2 = ImageRenderer(height=256, width=256)

    valid_location_xyz_low = (-2100.0, -1600.0, 30.0)
    valid_location_xyz_high = (2100.0, 1600.0, 200.0)

    valid_roll_pitch_yaw_low = (-10.0, -64.0, -180.0)
    valid_roll_pitch_yaw_high = (10, 30.0, 180.0)

    try:
        i = 0
        while True:
            if i % 10 == 0:
                randomize_level(http_bridge=bridge)

            location_xyz = tuple(np.random.uniform(low=valid_location_xyz_low, high=valid_location_xyz_high))
            roll_pitch_yaw = tuple(np.random.uniform(low=valid_roll_pitch_yaw_low, high=valid_roll_pitch_yaw_high))

            rgb_image, mask_image = bridge.capture_scene_at_location(
                location_xyz=location_xyz,
                roll_pitch_yaw=roll_pitch_yaw
            )
            if rgb_image is not None:
                # (Save rgb_image and mask_image here)

                image_renderer.render_cv2_image(cv2_image=rgb_image, channel_order="RGB")
                image_renderer2.render_cv2_image(cv2_image=mask_image, channel_order="RGB")
            i += 1

    except KeyboardInterrupt:
        print("stopping...")
    finally:
        pass
        image_renderer.close()
        image_renderer2.close()
        process_manger.close()
    print("done")
