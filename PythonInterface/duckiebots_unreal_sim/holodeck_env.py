import glob
from pathlib import Path
from typing import Optional, Type
import gym
import uuid

import numpy as np
from gym.spaces import Box
from holodeck.environments import *
from duckiebots_unreal_sim.observation_functions import UEDuckiebotsRGBCameraObservationFunction, \
    UEDuckiebotsSemanticMaskCameraObservationFunction, UEDuckiebotsRotAndLinearVelObservationFunction, UEDuckiebotsPositionAndYawObservationFunction
from duckiebots_unreal_sim.reward_functions import VelocityAndDistanceAlongTrackRewardAndTerminationFunction, UERewardAndTerminationFunction
from duckiebots_unreal_sim.tools import ImageRenderer
from duckiebots_unreal_sim.tools.process_manager import DEFAULT_GAME_LAUNCHER_PATH
from duckiebots_unreal_sim.rcan_obs_preprocessor import ONNXRCANObsPreprocessor

import cv2


class UEDuckiebotsHolodeckEnv:

    def __init__(self,
                 physics_hz: Optional[float] = 10.,
                 physics_ticks_between_action_and_observation: int = 1,
                 physics_ticks_between_observation_and_action: int = 0,
                 world_name="DuckiebotsHolodeckMap",
                 randomization_enabled: bool = True,
                 randomize_mask: bool = True,
                 render_game_on_screen: bool = False,
                 launch_game_process: bool = True,
                 return_rgb_and_mask_as_observation: bool = False,
                 use_rcan_instead_of_gt_mask: bool = False,
                 return_only_mask_as_observation: bool = False,
                 reward_function: Type[UERewardAndTerminationFunction] = VelocityAndDistanceAlongTrackRewardAndTerminationFunction,
                 randomize_every_n_steps: Optional[int] = None,
                 preprocess_rgb_with_rcan: bool = False,
                 image_obs_only: bool = False,
                 image_obs_out_height: int = 64,
                 image_obs_out_width: int = 64,
                 rcan_checkpoint_path: Optional[str] = None,
                 limit_backwards_movement: bool = False,
                 use_simple_physics: bool = False,
                 use_wheel_bias: bool = False,
                 randomize_camera_location_for_tilted_robot: bool = False,
                 game_path: Optional[str] = None):

        if sum([return_rgb_and_mask_as_observation, return_only_mask_as_observation, preprocess_rgb_with_rcan]) > 1:
            raise ValueError("Only one of return_rgb_and_mask_as_observation, "
                             "return_only_mask_as_observation, and "
                             "preprocess_rgb_with_rcan can be set to True.")

        self._physics_ticks_between_action_and_observation = physics_ticks_between_action_and_observation
        self._physics_ticks_between_observation_and_action = physics_ticks_between_observation_and_action

        self._randomization_enabled = randomization_enabled
        self._randomize_mask = randomize_mask
        self._randomize_camera_location_for_tilted_robot = randomize_camera_location_for_tilted_robot
        self._randomize_every_n_steps = randomize_every_n_steps
        self._steps_this_episode = 0

        self._return_rgb_and_mask_as_observation = return_rgb_and_mask_as_observation
        self._return_only_mask_as_observation = return_only_mask_as_observation

        self._use_rcan_instead_of_gt_mask = use_rcan_instead_of_gt_mask

        self._observation_out_height = image_obs_out_height
        self._observation_out_width = image_obs_out_width

        self._mask_observation_function = UEDuckiebotsSemanticMaskCameraObservationFunction(
            obs_out_height=self._observation_out_height,
            obs_out_width=self._observation_out_width)

        self._rgb_observation_function = UEDuckiebotsRGBCameraObservationFunction(
            obs_out_height=self._observation_out_height,
            obs_out_width=self._observation_out_width)

        self._velocity_observation_function = UEDuckiebotsRotAndLinearVelObservationFunction()
        self._position_and_yaw_observation_function = UEDuckiebotsPositionAndYawObservationFunction()

        self._reward_termination_function = reward_function()
        print(f"reward_termination_function: {self._reward_termination_function}")

        self._image_renderer = None
        self._renderer_image_scale = None
        self._mask_renderer = None

        self._shutdown = False
        self._has_finished_shutdown = False

        self.action_space = Box(low=-1.0, high=1.0, shape=(2,), dtype=np.float32)
        self.limit_backwards_movement = limit_backwards_movement
        self.use_wheel_bias = use_wheel_bias

        obs_space_channels = 3 if not self._return_rgb_and_mask_as_observation else 3 * 2
        self.observation_space = Box(low=0,
                                     high=255,
                                     shape=(self._observation_out_height, self._observation_out_width, obs_space_channels),
                                     dtype=np.uint8)
        if not image_obs_only:
            self.observation_space = {
                "image": self.observation_space,
                "yaw_and_forward_vel": Box(low=-np.inf,
                                           high=np.inf,
                                           shape=(2,),
                                           dtype=np.float32),
                "position_and_yaw": Box(low=-np.inf,
                                        high=np.inf,
                                        shape=(3,),
                                        dtype=np.float32)
            }

        self._individual_image_observation_space = Box(low=0,
                                                 high=255,
                                                 shape=(self._observation_out_height, self._observation_out_width, 3),
                                                 dtype=np.uint8)

        self._main_agent_name = "duckiebot_0"
        self._world_name = world_name

        holodeck_config = {
            "name": "duckiebots_holdeck",
            "world": self._world_name,
            "main_agent": self._main_agent_name,
            "agents": [
                {
                    "agent_name": "duckiebot_0",
                    "agent_type": "DuckiebotAgent",
                    "sensors": [
                        {
                            "sensor_type": "RGBCamera",
                            "sensor_name": "RGBCamera",
                            "existing": True,
                            "configuration": {
                                "CaptureHeight": 480,  # This should match the aspect ratio of the real duckiebot camera
                                "CaptureWidth": 640,
                            }
                        },
                        {
                            "sensor_type": "DuckiebotsSemanticMaskCamera",
                            "sensor_name": "DuckiebotsSemanticMaskCamera",
                            "existing": True,
                            "configuration": {
                                "CaptureHeight": 60,  # This should match the aspect ratio of the real duckiebot camera
                                "CaptureWidth": 80, 
                            }
                        },
                        {
                            "sensor_type": "DuckiebotsLineOverlapSensor",
                            "sensor_name": "DuckiebotsLineOverlapSensor",
                            "existing": True,
                        },
                        {
                            "sensor_type": "DuckiebotsVelocitySensor",
                            "sensor_name": "DuckiebotsVelocitySensor",
                            "existing": True,
                        },
                        {
                            "sensor_type": "ProgressAlongIntendedPathSensor",
                            "sensor_name": "ProgressAlongIntendedPathSensor",
                            "existing": True,
                        },
                        {
                            "sensor_type": "LocationSensor",
                            "sensor_name": "LocationSensor",
                            "existing": True,
                        },
                        {
                            "sensor_type": "YawSensor",
                            "sensor_name": "YawSensor",
                            "existing": True,
                        },
                    ],
                    "control_scheme": 1,
                    "location": [0, 0, 1],
                }
            ],
        }

        env_uuid = str(uuid.uuid4()) if launch_game_process else ""
        print(f"--launching env, uuid: {env_uuid} --")
        self.holodeck_env = HolodeckEnvironment(scenario=holodeck_config,
                                                binary_path=game_path or DEFAULT_GAME_LAUNCHER_PATH,
                                                start_world=launch_game_process,
                                                uuid=env_uuid,
                                                verbose=False,
                                                pre_start_steps=2,
                                                show_viewport=render_game_on_screen,
                                                ticks_per_sec=None if not physics_hz else physics_hz,
                                                copy_state=True,
                                                max_ticks=sys.maxsize)

        if not self._randomization_enabled:
            self.disable_movie_players()
        if use_simple_physics:
            self.enable_simple_physics()
        print("--env launch complete--")

        self._preprocess_rgb_with_rcan = preprocess_rgb_with_rcan
        self._rcan = None
        if self._preprocess_rgb_with_rcan or self._use_rcan_instead_of_gt_mask:
            print("--initializing RCAN--")
            if not rcan_checkpoint_path:
                raise ValueError("rcan_checkpoint_path must be specified if preprocess_rgb_with_rcan is set to True.")
            self._rcan = ONNXRCANObsPreprocessor(checkpoint_path=rcan_checkpoint_path, debug_render_predictions=False)
            print("--RCAN set up--")

    def reset(self, *args, **kwargs):
        self._steps_this_episode = 0

        if self._randomization_enabled:
            self.randomize(randomize_mask=self._randomize_mask)
        self._reward_termination_function.reset()
        self._latest_observation_state = self.holodeck_env.reset(load_new_agents=False)

        return self._get_obs()

    def _get_obs(self):
        if self._return_rgb_and_mask_as_observation:
            rgb_obs = self._rgb_observation_function.get_observation_for_timestep(
                holodeck_state=self._latest_observation_state)
            assert rgb_obs in self._individual_image_observation_space, (rgb_obs.shape, self._individual_image_observation_space)

            if self._use_rcan_instead_of_gt_mask:
                mask_obs = self._rcan.preprocess_obs(rgb_obs=rgb_obs)
            else:
                mask_obs = self._mask_observation_function.get_observation_for_timestep(
                    holodeck_state=self._latest_observation_state)
            # mask_obs = self._rcan.preprocess_obs(rgb_obs=rgb_obs)
            assert mask_obs in self._individual_image_observation_space, (mask_obs.shape, self._individual_image_observation_space)
            obs = np.concatenate((rgb_obs, mask_obs), axis=-1)

        elif self._return_only_mask_as_observation:
            if self._use_rcan_instead_of_gt_mask:
                mask_obs = self._rcan.preprocess_obs(rgb_obs=rgb_obs)
            else:
                mask_obs = self._mask_observation_function.get_observation_for_timestep(
                    holodeck_state=self._latest_observation_state)
            assert mask_obs in self._individual_image_observation_space, (mask_obs.shape, self._individual_image_observation_space)
            obs = mask_obs
        else:
            rgb_obs = self._rgb_observation_function.get_observation_for_timestep(
                holodeck_state=self._latest_observation_state)

            if self._preprocess_rgb_with_rcan:
                rgb_obs = self._rcan.preprocess_obs(rgb_obs=rgb_obs)
            assert rgb_obs in self._individual_image_observation_space, (rgb_obs.shape, rgb_obs.dtype, self._individual_image_observation_space)
            obs = rgb_obs

        if isinstance(self.observation_space, dict):
            obs = {
                "image": obs,
                "yaw_and_forward_vel": self._velocity_observation_function.get_observation_for_timestep(
                    holodeck_state=self._latest_observation_state),
                "position_and_yaw": self._position_and_yaw_observation_function.get_observation_for_timestep(
                    holodeck_state=self._latest_observation_state)
            }
        return obs

    def get_duckiebot_state(self) -> np.ndarray:
        location_x = self._latest_observation_state['LocationSensor'][0]
        location_y = self._latest_observation_state['LocationSensor'][1]
        yaw = self._latest_observation_state['YawSensor'][0]
        forward_velocity = self._latest_observation_state['DuckiebotsVelocitySensor'][0]
        yaw_velocity = self._latest_observation_state['DuckiebotsVelocitySensor'][1]

        return np.asarray([location_x, location_y, yaw, forward_velocity, yaw_velocity], dtype=np.float32)

    def get_duckiebot_metrics_info(self):
        return {
            "progress_along_intended_path": self._latest_observation_state["ProgressAlongIntendedPathSensor"][0],
            "distance_cm_from_path_center": self._latest_observation_state["ProgressAlongIntendedPathSensor"][2] * 0.1,
            "velocity_along_intended_path": self._latest_observation_state["ProgressAlongIntendedPathSensor"][1],
        }

    def step(self, action: np.ndarray, override_duckiebot_state=None):
        if action not in self.action_space:
            raise ValueError(f"action {action} not in action_space {self.action_space}")

        if self.limit_backwards_movement:
            action[0] = max(-0.2, float(action[0]))

        if self.use_wheel_bias:
            orig_yaw_action = action[1]
            if orig_yaw_action > 0.9:
                new_yaw_action = ((orig_yaw_action - 0.9) / 2.0) + 0.9
            else:
                new_yaw_action = orig_yaw_action - 0.3
                if orig_yaw_action < 0.0:
                    new_yaw_action *= 2.0

            action[1] = new_yaw_action

        holodeck_states_this_step = []

        action_buffer = np.zeros(shape=(8,), dtype=np.float32)
        if override_duckiebot_state is not None:
            # print(f"override_duckiebot_state: {override_duckiebot_state}")
            action_buffer[2] = 1.0
            # action_buffer[3:] = override_duckiebot_state  # ResetX, ResetY, ResetYaw, ResetForwardVelocity, ResetYawVelocity

            action_buffer[3] = override_duckiebot_state[0]
            action_buffer[4] = override_duckiebot_state[1]
            action_buffer[5] = override_duckiebot_state[2] # not including velocity in reset
            self.holodeck_env.act(agent_name=self._main_agent_name, action=action_buffer)
            self._latest_observation_state = self.holodeck_env.tick(num_ticks=1)
            action_buffer = np.zeros(shape=(8,), dtype=np.float32)

        action_buffer[:2] = action
        # print(f"Sending action buffer: {action_buffer}")
        self.holodeck_env.act(agent_name=self._main_agent_name, action=action_buffer)
        for _ in range(self._physics_ticks_between_action_and_observation):
            self._latest_observation_state = self.holodeck_env.tick(num_ticks=1)

            # action_buffer = np.zeros(shape=(8,), dtype=np.float32)
            # action_buffer[:2] = action
            # self.holodeck_env.act(agent_name=self._main_agent_name, action=action_buffer)

            holodeck_states_this_step.append(self._latest_observation_state)

        obs = self._get_obs()

        for _ in range(self._physics_ticks_between_observation_and_action):
            latest_state = self.holodeck_env.tick(num_ticks=1)
            holodeck_states_this_step.append(latest_state)

        reward, done = self._reward_termination_function.get_reward_and_done_for_current_timestep(
            last_action=action,
            holodeck_states_this_step=holodeck_states_this_step
        )

        self._steps_this_episode += 1

        if self._randomization_enabled and self._randomize_every_n_steps and \
                self._steps_this_episode % self._randomize_every_n_steps == 0:
            self.randomize(randomize_mask=self._randomize_mask)

        return obs, reward, done, {}

    def render(self, mode="human", image_scale=1.0, only_show_rgb_if_combined_obs=False):
        if self._return_rgb_and_mask_as_observation:
            obs = self._get_obs()
            if isinstance(obs, dict):
                obs = obs['image']
            (rgb_obs, mask_obs) = np.split(obs, indices_or_sections=2, axis=-1)
            if only_show_rgb_if_combined_obs:
                image_to_render_rgb_hwc = rgb_obs
            else:
                image_to_render_rgb_hwc = np.concatenate((rgb_obs, mask_obs), axis=1)
        else:
            image_to_render_rgb_hwc = self._get_obs()
            if isinstance(image_to_render_rgb_hwc, dict):
                image_to_render_rgb_hwc = image_to_render_rgb_hwc['image']

        if int(image_scale) > 1.0:
            orig_height = image_to_render_rgb_hwc.shape[0]
            orig_width = image_to_render_rgb_hwc.shape[1]
            image_to_render_rgb_hwc = cv2.resize(image_to_render_rgb_hwc,
                                                 (orig_width * int(image_scale), orig_height * int(image_scale)),
                                                 cv2.INTER_NEAREST_EXACT)

        if image_scale != self._renderer_image_scale:
            self._renderer_image_scale = image_scale
            if self._image_renderer:
                self._image_renderer.close()
                self._image_renderer = None

        if mode == "human":
            if not self._image_renderer:
                height = image_to_render_rgb_hwc.shape[0]
                width = image_to_render_rgb_hwc.shape[1]
                print(f"input: {(width, height)}")
                self._image_renderer = ImageRenderer(height=height, width=width)

            # print(f"renderer: {(self._image_renderer._width, self._image_renderer._height)}")
            self._image_renderer.render_cv2_image(image_to_render_rgb_hwc)
        elif mode == "rgb_array":
            return image_to_render_rgb_hwc
        else:
            raise ValueError(f"Unknown render mode: {mode}")

    def render_mask(self, mode="human", image_scale=1.0):
        color_image_to_render_rgb_hwc: np.ndarray = self._latest_observation_state["DuckiebotsSemanticMaskCamera"][::-1,
                                                    :, :3]
        if mode == "human":
            if color_image_to_render_rgb_hwc is not None:
                if int(image_scale) > 1.0:
                    orig_height = color_image_to_render_rgb_hwc.shape[0]
                    orig_width = color_image_to_render_rgb_hwc.shape[1]
                    color_image_to_render_rgb_hwc = cv2.resize(color_image_to_render_rgb_hwc,
                                                         (
                                                         orig_width * int(image_scale), orig_height * int(image_scale)),
                                                         cv2.INTER_NEAREST_EXACT)
                    print(f"image_scale: {int(image_scale)}: {(orig_width, orig_height)} -> {color_image_to_render_rgb_hwc.shape})")

                if image_scale != self._renderer_image_scale:
                    self._renderer_image_scale = image_scale
                    if self._mask_renderer:
                        self._mask_renderer.close()
                        self._mask_renderer = None

                if not self._mask_renderer:
                    height = color_image_to_render_rgb_hwc.shape[0]
                    width = color_image_to_render_rgb_hwc.shape[1]
                    self._mask_renderer = ImageRenderer(height=height, width=width)

                self._mask_renderer.render_cv2_image(color_image_to_render_rgb_hwc)
        elif mode == "rgb_array":
            return color_image_to_render_rgb_hwc
        else:
            raise ValueError(f"Unknown render mode: {mode}")

    def enable_simple_physics(self):
        self.holodeck_env.send_world_command(name="EnableSimplePhysics")

    def randomize(self, randomize_mask: bool = None, randomize_camera_location_for_tilted_robot: bool = None):
        if randomize_mask is None:
            randomize_mask = self._randomize_mask

        if randomize_camera_location_for_tilted_robot is None:
            randomize_camera_location_for_tilted_robot = self._randomize_camera_location_for_tilted_robot

        random_backdrop_movie_path = ""
        random_road_movie_path = ""

        is_domain_randomization_world = (self._world_name == "DuckiebotsHolodeckMapDomainRandomization")
        randomize_world_appearance = is_domain_randomization_world

        # movie_file_type = "bk2"
        # backdrop_movie_directory = os.path.join(os.path.dirname(os.path.abspath(__file__)), "backdrop_movies")
        # if not os.path.isdir(backdrop_movie_directory):
        #     raise NotADirectoryError(f"{backdrop_movie_directory} isn't a directory.")
        # backdrop_movie_file_paths = glob.glob(f"{backdrop_movie_directory}/*.{movie_file_type}")
        # if len(backdrop_movie_file_paths) > 0:
        #     random_backdrop_movie_path = Path(random.choice(backdrop_movie_file_paths)).as_posix()
        #
        # road_movie_directory = os.path.join(os.path.dirname(os.path.abspath(__file__)), "road_movies")
        # if not os.path.isdir(road_movie_directory):
        #     raise NotADirectoryError(f"{road_movie_directory} isn't a directory.")
        # road_movie_file_paths = glob.glob(f"{road_movie_directory}/*.{movie_file_type}")
        # if len(road_movie_file_paths) > 0:
        #     random_road_movie_path = Path(random.choice(road_movie_file_paths)).as_posix()
        # print(f"movie paths are\n{random_backdrop_movie_path}\n{random_road_movie_path}")

        disable_movie_player = False

        self.holodeck_env.send_world_command(
            name="RandomizeDuckiebotsWorld",
            num_params=[float(randomize_mask),
                        float(disable_movie_player),
                        float(randomize_camera_location_for_tilted_robot),
                        float(randomize_world_appearance)],
            string_params=[random_backdrop_movie_path,
                           random_road_movie_path])

    def randomize_physics(self,
                          relative_vel_scale: float = 1.0,
                          relative_turn_scale: float = 1.0,
                          cam_x_offset: float = 0.0,
                          cam_y_offset: float = 0.0,
                          cam_z_offset: float = 0.0,
                          cam_roll_offset: float = 0.0,
                          cam_pitch_offset: float = 0.0,
                          cam_yaw_offset: float = 0.0):
        self.holodeck_env.send_world_command(
            name="SetPhysicsRandomization",
            num_params=[float(relative_vel_scale),
                        float(relative_turn_scale),
                        float(cam_x_offset),
                        float(cam_y_offset),
                        float(cam_z_offset),
                        float(cam_roll_offset),
                        float(cam_pitch_offset),
                        float(cam_yaw_offset),],
            string_params=[])

    def disable_movie_players(self):
        random_backdrop_movie_path = ""
        random_road_movie_path = ""
        randomize_mask = False
        disable_movie_player = True
        randomize_camera_location_for_tilted_robot = False
        self.holodeck_env.send_world_command(
            name="RandomizeDuckiebotsWorld",
            num_params=[float(randomize_mask), float(disable_movie_player), float(randomize_camera_location_for_tilted_robot)],
            string_params=[random_backdrop_movie_path, random_road_movie_path])

    def close(self):
        self.holodeck_env.__on_exit__()
        if self._image_renderer:
            self._image_renderer.close()
        if self._mask_renderer:
            self._mask_renderer.close()

    def __enter__(self):
        self.holodeck_env.__enter__()
        return super().__enter__()

    def __exit__(self, *args):
        self.close()
        return super().__exit__(*args)

    def __del__(self):
        self.close()


if __name__ == '__main__':
    from tools import XboxController, get_keyboard_turning, get_keyboard_velocity

    print("Detecting gamepad (you may have to press a button on the controller)...")
    gamepad = None
    if XboxController.detect_gamepad():
        gamepad = XboxController()
    print("Gamepad found" if gamepad else "use keyboard controls")

    # Change this to False in order to attach to a standalone game launched in the UE editor
    # True launches a game stored in the packaged_games directory and attaches to it.
    launch_game_process = True

    # Change this to False in order to not render the video games primary view (showing spectator mode).
    render_launched_game_on_screen = True
    game_launcher_path = DEFAULT_GAME_LAUNCHER_PATH
    locked_physics_rate = True

    env = UEDuckiebotsHolodeckEnv(launch_game_process=launch_game_process,
                                  render_game_on_screen=render_launched_game_on_screen,
                                  randomization_enabled=False,
                                  physics_hz=30 if locked_physics_rate else None)

    print("env initialized")
    while True:
        i = 0
        env.reset()
        env.render()
        # env.render_mask()

        # color_image = env.render(mode="rgb_array")
        # mask_image = env.render_mask(mode="rgb_array")
        episode_reward = 0.0
        done = False
        while not done:
            # input("press enter to progress")

            velocity = gamepad.LeftJoystickY if gamepad else get_keyboard_velocity()
            turning = gamepad.LeftJoystickX if gamepad else get_keyboard_turning()
            action = np.asarray([velocity, turning], np.float32)
            print(f"action: {action}")
            # Example to randomly select an action in the environment:
            # action = env.action_space.sample()
            # print(f"duckiebot state: {env.get_duckiebot_state()}")
            env.get_duckiebot_state()
            obs, rew, done, info = env.step(action=action)
            # print(f"duckiebot state: {env.get_duckiebot_state()}")
            # print(f"obs state: {env._latest_observation_state}")

            episode_reward += rew

            env.render()
            # env.render_mask()
            # color_image = env.render(mode="rgb_array")
            # mask_image = env.render_mask(mode="rgb_array")

            if gamepad and gamepad.B:
                print("randomizing")
                # env.reset()
                env.randomize()

            if gamepad and gamepad.Y:
                print("manual reset")
                env.reset()

            i += 1
        print(f"episode reward: {episode_reward}")
