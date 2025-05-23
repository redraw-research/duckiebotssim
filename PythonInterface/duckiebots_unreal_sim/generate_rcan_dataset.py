from duckiebots_unreal_sim.holodeck_lane_following_env import UELaneFollowingEnv
import numpy as np
import time

if __name__ == '__main__':
    from duckiebots_unreal_sim.tools import XboxController, get_keyboard_turning, get_keyboard_velocity
    from duckiebots_unreal_sim.tools import ensure_dir
    import tqdm
    # print("Detecting gamepad (you may have to press a button on the controller)...")
    # gamepad = None
    # if XboxController.detect_gamepad():
    #     gamepad = XboxController()
    # print("Gamepad found" if gamepad else "use keyboard controls")

    env = UELaneFollowingEnv({
        "use_domain_randomization": True,
        "render_game_on_screen": False,
        "use_mask_observation": False,
        "return_rgb_and_mask_as_observation": True,
        "preprocess_rgb_observations_with_rcan": False,
        "rcan_checkpoint_path": "/home/author1/Downloads/ckpt-91.onnx",
        "launch_game_process": True,
        "simulate_latency": True,
        "normalize_image": False,
        "frame_stack_amount": 1,
        "physics_hz": 20,
        "use_simple_physics": False,
        "randomize_camera_location_for_tilted_robot": True,
        "world_name": "DuckiebotsHolodeckMapDomainRandomization",
    })

    print("env initialized")

    data_gen_run_name = "new_track_v1"
    num_images_to_generate = int(200e3)
    progress_bar = tqdm.tqdm(total=num_images_to_generate)
    images_generated = 0

    render_images_with_scale = 10.0
    # target_frame_time = 1.0/10.0
    while True:
        env.reset()
        env.render(image_scale=render_images_with_scale)
        done = False
        prev_frame_time = time.time()
        episode_steps = 0
        while not done and episode_steps < 200:
            # velocity = gamepad.LeftJoystickY if gamepad else get_keyboard_velocity()
            # turning = gamepad.LeftJoystickX if gamepad else get_keyboard_turning()
            # action = np.asarray([velocity, turning], np.float32)

            action = env.action_space.sample()

            obs, rew, done, info = env.step(action=action)

            # current_delta = time.time() - prev_frame_time
            # sleep_delta = max(target_frame_time - current_delta, 0)
            # time.sleep(sleep_delta)
            # now = time.time()
            # print(f"delta time: {now - prev_frame_time}")
            # prev_frame_time = now

            env.render(image_scale=render_images_with_scale)

            rgb_image = obs['image'][:, :, :3]  # unprocessed rgb image
            mask_image = obs['image'][:, :, 3:]  # semantic mask image (in RGB color space)

            assert tuple(rgb_image.shape) == (64, 64, 3), rgb_image.shape
            assert tuple(mask_image.shape) == (64, 64, 3), mask_image.shape
            assert np.min(rgb_image) >= 0
            assert np.max(rgb_image) <= 255
            assert np.min(mask_image) >= 0
            assert np.max(mask_image) <= 255

            color_npy_save_path = f'/home/author1/Downloads/rcan_data_nov_16/frame/{data_gen_run_name}_f{images_generated}'
            segment_npy_save_path = f'/home/author1/Downloads/rcan_data_nov_16/segment/{data_gen_run_name}_f{images_generated}'

            ensure_dir(color_npy_save_path)
            ensure_dir(segment_npy_save_path)
            np.save(color_npy_save_path, rgb_image[:, :, ::-1])
            np.save(segment_npy_save_path, mask_image[:, :, ::-1])
            images_generated += 1
            progress_bar.update(1)
            if images_generated >= num_images_to_generate:
                print(f"Stopping. Generated {images_generated} images.")
                exit(0)

            if done:
                # print("Done")
                env.reset()

            # if gamepad and gamepad.B:
            #     print("randomizing")
            #     env.base_env.randomize()
            #
            # if gamepad and gamepad.Y:
            #     print("manual reset")
            #     env.reset()

            episode_steps += 1