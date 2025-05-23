import time

import gym
import numpy as np
from duckiebots_unreal_sim.holodeck_env import UEDuckiebotsHolodeckEnv
from gym import spaces


class NormalizeWrapper(gym.ObservationWrapper):
    def __init__(self, env=None):
        super(NormalizeWrapper, self).__init__(env)
        self.obs_lo = self.observation_space.low[0, 0, 0]
        self.obs_hi = self.observation_space.high[0, 0, 0]
        obs_shape = self.observation_space.shape
        self.observation_space = gym.spaces.Box(0.0, 1.0, obs_shape, dtype=np.float32)

    def step(self, action):
        observation, reward, terminated, info = self.env.step(action)
        return self.observation(observation), reward, terminated, info

    def reset(self):
        """Resets the environment, returning a modified observation using :meth:`self.observation`."""
        obs = self.env.reset()
        return self.observation(obs)

    def observation(self, obs):
        if self.obs_lo == 0.0 and self.obs_hi == 1.0:
            return obs
        else:
            return (obs - self.obs_lo) / (self.obs_hi - self.obs_lo)


class ObservationBufferWrapper(gym.ObservationWrapper):
    def __init__(self, env=None, obs_buffer_depth=3):
        super(ObservationBufferWrapper, self).__init__(env)
        obs_space_shape_list = list(self.observation_space.shape)

        # The last dimension, is used. For images, this should be the depth.
        # For vectors, the output is still a vector, just concatenated.
        self.buffer_axis = len(obs_space_shape_list) - 1
        obs_space_shape_list[self.buffer_axis] *= obs_buffer_depth
        # self.observation_space.shape = tuple(obs_space_shape_list)

        if len(self.observation_space.shape) == 3:
            limit_low = self.observation_space.low[0, 0, 0]
            limit_high = self.observation_space.high[0, 0, 0]
        elif len(self.observation_space.shape) == 1:
            # Note this was implemented for vector like observation spaces (e.g. a VAE latent vector)
            limit_low = self.observation_space.low[0]
            limit_high = self.observation_space.high[0]
        else:
            assert False, "Only 1 or 3 dimentsional obs space supported!"

        self.observation_space = spaces.Box(
            limit_low,
            limit_high,
            tuple(obs_space_shape_list),
            dtype=self.observation_space.dtype)
        self.obs_buffer_depth = obs_buffer_depth
        self.obs_buffer = None

    def observation(self, obs):
        if self.obs_buffer_depth == 1:
            return obs
        if self.obs_buffer is None:
            self.obs_buffer = np.concatenate([obs for _ in range(self.obs_buffer_depth)], axis=self.buffer_axis,
                                             dtype=self.observation_space.dtype)
        else:
            self.obs_buffer = np.concatenate((self.obs_buffer[..., (obs.shape[self.buffer_axis]):], obs),
                                             axis=self.buffer_axis, dtype=self.observation_space.dtype)
        return self.obs_buffer

    def step(self, action):
        observation, reward, terminated, info = self.env.step(action)
        return self.observation(observation), reward, terminated, info

    def reset(self):
        self.obs_buffer = None
        observation = self.env.reset()
        return self.observation(observation)


class UELaneFollowingEnv(gym.Env):
    def __init__(self, config: dict = None):
        env_config = {
            "use_domain_randomization": True,
            "randomize_mask": False,
            "render_game_on_screen": False,
            "use_mask_observation": False,
            "return_rgb_and_mask_as_observation": False,
            "simulate_latency": True,
            "preprocess_rgb_observations_with_rcan": False,
            "rcan_checkpoint_path": None,
            "frame_stack_amount": 2,
            "normalize_image": False,
            "launch_game_process": True,
            "physics_hz": None,
            "limit_backwards_movement": False,
            "use_simple_physics": False,
            "use_wheel_bias": False,
            "randomize_camera_location_for_tilted_robot": False,
            "world_name": "DuckiebotsHolodeckMap",
            "image_obs_out_height": 64,
            "image_obs_out_width": 64,
            "randomize_physics_every_step": False
        }

        env_config.update(config)

        physics_hz = env_config['physics_hz'] or (6*2 if env_config["simulate_latency"] else 6)

        self.base_env = UEDuckiebotsHolodeckEnv(
            randomization_enabled=env_config["use_domain_randomization"],
            physics_hz=physics_hz,
            physics_ticks_between_action_and_observation=1,
            physics_ticks_between_observation_and_action=1 if env_config["simulate_latency"] else 0,
            render_game_on_screen=env_config["render_game_on_screen"],
            return_only_mask_as_observation=env_config["use_mask_observation"],
            return_rgb_and_mask_as_observation=env_config["return_rgb_and_mask_as_observation"],
            randomize_mask=env_config["randomize_mask"],
            preprocess_rgb_with_rcan=env_config["preprocess_rgb_observations_with_rcan"],
            rcan_checkpoint_path=env_config["rcan_checkpoint_path"],
            image_obs_only=False,
            launch_game_process=env_config["launch_game_process"],
            limit_backwards_movement=env_config['limit_backwards_movement'],
            use_simple_physics=env_config['use_simple_physics'],
            use_wheel_bias=env_config['use_wheel_bias'],
            randomize_camera_location_for_tilted_robot=env_config['randomize_camera_location_for_tilted_robot'],
            world_name=env_config["world_name"],
            image_obs_out_height=env_config["image_obs_out_height"],
            image_obs_out_width=env_config["image_obs_out_width"],
        )

        # Observation Wrappers
        self._env: gym.Env = self.base_env

        if env_config["normalize_image"]:
            self._env = NormalizeWrapper(self._env)

        if env_config["frame_stack_amount"] > 1:
            self._env = ObservationBufferWrapper(self._env, obs_buffer_depth=env_config["frame_stack_amount"])

        self._randomize_physics_every_step = env_config['randomize_physics_every_step']

        self.action_space = self._env.action_space
        self.observation_space = self._env.observation_space

    def render(self, mode='human', image_scale=1.0, only_show_rgb_if_combined_obs=False):
        return self.base_env.render(mode=mode, image_scale=image_scale,
                                    only_show_rgb_if_combined_obs=only_show_rgb_if_combined_obs)

    def reset(self):
        return self._env.reset()

    def step(self, action):
        if self._randomize_physics_every_step:
            self.base_env.randomize_physics(relative_vel_scale=1.0 + (np.random.standard_normal() / 3.0),
                                            relative_turn_scale=1.0 + (np.random.standard_normal() / 3.0),
                                            cam_x_offset=np.random.uniform(low=-10.0, high=10.0),
                                            cam_y_offset=np.random.uniform(low=-1.0, high=1.0),
                                            cam_z_offset=np.random.uniform(low=-10.0, high=10.0),
                                            cam_roll_offset=np.random.uniform(low=-2.0, high=2.0),
                                            cam_pitch_offset=np.random.uniform(low=-5.0, high=5.0),
                                            cam_yaw_offset=np.random.uniform(low=-0.3, high=0.3),
                                            )

        s, r, d, info = self._env.step(action)
        if not isinstance(s, dict):
            assert s in self.observation_space, f"observation space: {self.observation_space}, " \
                                                f"observation shape: {s.shape} " \
                                                f"observation_dtype: {s.dtype}"
        return s, r, d, info

    def close(self):
        print("closing UELaneFollowingTask")
        self.base_env.close()


if __name__ == '__main__':
    from duckiebots_unreal_sim.tools import XboxController, get_keyboard_turning, get_keyboard_velocity

    print("Detecting gamepad (you may have to press a button on the controller)...")
    gamepad = None
    if XboxController.detect_gamepad():
        gamepad = XboxController()
    print("Gamepad found" if gamepad else "use keyboard controls")

    env = UELaneFollowingEnv({
        "use_domain_randomization": False,
        "render_game_on_screen": True,
        "use_mask_observation": False,
        "return_rgb_and_mask_as_observation": False,
        # "preprocess_rgb_observations_with_rcan": True,
        # "rcan_checkpoint_path": "/home/author1/Downloads/ckpt-91.onnx",
        "rcan_checkpoint_path": "/home/author1/Downloads/ckpt_9_nov_17.onnx",
        "launch_game_process": True,
        "simulate_latency": True,
        "normalize_image": False,
        "frame_stack_amount": 1,
        "physics_hz": 20,
        "use_simple_physics": False,
        "randomize_camera_location_for_tilted_robot": False,
        "world_name": "DuckiebotsHolodeckMapDomainRandomization",
        "randomize_physics_every_step": False
    })

    print("env initialized")


    render_images_with_scale = 10.0
    # target_frame_time = 1.0/10.0
    while True:
        env.reset()
        env.render(image_scale=render_images_with_scale)
        done = False
        prev_frame_time = time.time()
        episode_steps = 0
        while not done and episode_steps < 200:
            velocity = gamepad.LeftJoystickY if gamepad else get_keyboard_velocity()
            turning = gamepad.LeftJoystickX if gamepad else get_keyboard_turning()
            action = np.asarray([velocity, turning], np.float32)

            # action = env.action_space.sample()

            obs, rew, done, info = env.step(action=action)

            # current_delta = time.time() - prev_frame_time
            # sleep_delta = max(target_frame_time - current_delta, 0)
            # time.sleep(sleep_delta)
            # now = time.time()
            # print(f"delta time: {now - prev_frame_time}")
            # prev_frame_time = now

            env.render(image_scale=render_images_with_scale)

            if done:
                print("Done")
                env.reset()

            if gamepad and gamepad.B:
                print("randomizing")
                env.base_env.randomize()
            #
            if gamepad and gamepad.Y:
                print("manual reset")
                env.reset()

            episode_steps += 1