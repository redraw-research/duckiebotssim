import sys
import time
from typing import Optional, Tuple, Union, List
from pathlib import Path
import cv2
import gym
import numpy as np
from gym.core import RenderFrame, ObsType, ActType
from gym.spaces import Box
import uuid
import holodeck
from holodeck import agents
from holodeck.environments import *
from holodeck import sensors
import glob

from duckiebots_unreal_sim.level_randomization import randomize_level
from duckiebots_unreal_sim.reward_functions import UERewardAndTerminationFunction, \
    DefaultDuckiebotsRewardAndTerminationFunction
from duckiebots_unreal_sim.tools import UEHTTPBridge, UEProcessManager, ImageRenderer, NDIImageReceiver, Rate, \
    DEFAULT_GAME_LAUNCHER_PATH
from duckiebots_unreal_sim.observation_functions import UEDuckiebotsRGBCameraObservationFunction, UEDuckiebotsSemanticMaskCameraObservationFunction
from duckiebots_unreal_sim.tools.process_manager import DEFAULT_GAME_LAUNCHER_PATH
from duckiebots_unreal_sim.tools.util import ensure_dir

class UEDuckiebotsHolodeckEnv(gym.Env):

    def __init__(self,
                 physics_hz: Optional[float] = 10.,
                 physics_ticks_between_action_and_observation: int = 1,
                 physics_ticks_between_observation_and_action: int = 1,
                 randomization_enabled: bool = True,
                 randomize_movies: bool = True,
                 render_game_on_screen: bool = False,
                 launch_game_process: bool = True):

        self._physics_ticks_between_action_and_observation = physics_ticks_between_action_and_observation
        self._physics_ticks_between_observation_and_action = physics_ticks_between_observation_and_action

        self._randomize_movies = randomize_movies
        self._randomization_enabled = randomization_enabled

        self._observation_out_height = 60
        self._observation_out_width = 80


        self._observation_function = UEDuckiebotsRGBCameraObservationFunction(obs_out_height=self._observation_out_height,
                                                                              obs_out_width=self._observation_out_width)

        # self._observation_function = UEDuckiebotsSemanticMaskCameraObservationFunction(obs_out_height=self._observation_out_height,
        #                                                                       obs_out_width=self._observation_out_width)

        self._reward_termination_function = DefaultDuckiebotsRewardAndTerminationFunction(
            max_episode_length=10000,
            drive_off_road_penalty_magnitude=10.0,
            wrong_way_penalty_magnitude=0.1,
            visit_new_tile_reward=5.0
        )

        self._image_renderer = None
        self._mask_renderer = None

        self._shutdown = False
        self._has_finished_shutdown = False

        self.action_space = Box(low=-1.0, high=1.0, shape=(2,), dtype=np.float32)
        self.observation_space = Box(low=0,
                                     high=256,
                                     shape=(self._observation_out_height, self._observation_out_width,  3),
                                     dtype=np.float32)

        self._main_agent_name = "duckiebot_0"

        holodeck_config = {
            "name": "duckiebots_holdeck",
            "world": "DuckiebotsHolodeckMap",
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
                                "CaptureHeight": 120,
                                "CaptureWidth": 160,
                            }
                        },
                        {
                            "sensor_type": "DuckiebotsSemanticMaskCamera",
                            "sensor_name": "DuckiebotsSemanticMaskCamera",
                            "existing": True,
                            "configuration": {
                                "CaptureHeight": 60,
                                "CaptureWidth": 80,
                            }
                        },
                        {
                            "sensor_type": "DuckiebotsLoopStatusSensor",
                            "sensor_name": "DuckiebotsLoopStatusSensor",
                            "existing": True,
                        },
                    ],
                    "control_scheme": 1,
                    "location": [0, 0, 1],
                }
            ],
        }

        print("--launching env--")
        self.holodeck_env = HolodeckEnvironment(scenario=holodeck_config,
                                                binary_path=DEFAULT_GAME_LAUNCHER_PATH,
                                                start_world=launch_game_process,
                                                uuid=str(uuid.uuid4()) if launch_game_process else "",
                                                verbose=True,
                                                pre_start_steps=2,
                                                show_viewport=render_game_on_screen,
                                                ticks_per_sec=None if not physics_hz else physics_hz,
                                                copy_state=True,
                                                max_ticks=sys.maxsize)
        print("--env launch complete--")

    def reset(self, *args, **kwargs) -> ObsType:
        if self._randomization_enabled:
            self.randomize()
        self._reward_termination_function.reset()
        self._latest_observation_state = self.holodeck_env.reset(load_new_agents=False)
        obs = self._observation_function.get_observation_for_timestep(holodeck_state=self._latest_observation_state)
        assert obs in self.observation_space, (obs.shape, self.observation_space)
        return obs

    def step(self, action: ActType) -> Tuple[ObsType, float, bool, dict]:
        if action not in self.action_space:
            raise ValueError(f"action {action} not in action_space {self.action_space}")

        holodeck_states_this_step = []

        self.holodeck_env.act(agent_name=self._main_agent_name, action=action)
        for _ in range(self._physics_ticks_between_action_and_observation):
            self._latest_observation_state = self.holodeck_env.tick(num_ticks=1)
            holodeck_states_this_step.append(self._latest_observation_state)

        # if self._latest_observation_state["DuckiebotsLoopStatusSensor"][0]:
        #     print(f"hit white line")
        #
        # if self._latest_observation_state["DuckiebotsLoopStatusSensor"][1]:
        #     print(f"hit yellow line")
        #
        # if self._latest_observation_state["DuckiebotsLoopStatusSensor"][3]:
        #     print(f"entered new road tile")

        obs = self._observation_function.get_observation_for_timestep(holodeck_state=self._latest_observation_state)
        assert obs in self.observation_space, (obs.shape, self.observation_space)

        for _ in range(self._physics_ticks_between_observation_and_action):
            latest_state = self.holodeck_env.tick(num_ticks=1)
            holodeck_states_this_step.append(latest_state)

        reward, done = self._reward_termination_function.get_reward_and_done_for_current_timestep(
            holodeck_states_this_step=holodeck_states_this_step
        )

        return obs, reward, done, {}

    def render(self, mode="human") -> Optional[Union[RenderFrame, List[RenderFrame]]]:
        image_to_render_bgr_hwc = self._observation_function.get_observation_for_timestep(
            holodeck_state=self._latest_observation_state)
        if mode == "human":
            if not self._image_renderer:
                height = image_to_render_bgr_hwc.shape[0]
                width = image_to_render_bgr_hwc.shape[1]
                self._image_renderer = ImageRenderer(height=height, width=width)
            if image_to_render_bgr_hwc is not None:
                self._image_renderer.render_cv2_image(image_to_render_bgr_hwc)
        elif mode == "rgb_array":
            return image_to_render_bgr_hwc
        else:
            raise ValueError(f"Unknown render mode: {mode}")

    def render_mask(self, mode="human") -> Optional[Union[RenderFrame, List[RenderFrame]]]:
        color_image_to_render_bgr_hwc: np.ndarray = self._latest_observation_state["DuckiebotsSemanticMaskCamera"][::-1, :, :3]
        if mode == "human":
            if not self._mask_renderer:
                height = color_image_to_render_bgr_hwc.shape[0]
                width = color_image_to_render_bgr_hwc.shape[1]
                self._mask_renderer = ImageRenderer(height=height, width=width)
            if color_image_to_render_bgr_hwc is not None:
                self._mask_renderer.render_cv2_image(color_image_to_render_bgr_hwc)
        elif mode == "rgb_array":
            return color_image_to_render_bgr_hwc
        else:
            raise ValueError(f"Unknown render mode: {mode}")

    def randomize(self, randomize_mask: bool = True):
        random_backdrop_movie_path = ""
        random_road_movie_path = ""

        # if randomize_movies:
        #     backdrop_movie_directory = os.path.join(os.path.dirname(os.path.abspath(__file__)), "backdrop_movies")
        #     if not os.path.isdir(backdrop_movie_directory):
        #         raise NotADirectoryError(f"{backdrop_movie_directory} isn't a directory.")
        #     backdrop_movie_file_paths = glob.glob(f"{backdrop_movie_directory}/*.{movie_file_type}")
        #     if len(backdrop_movie_file_paths) > 0:
        #         random_backdrop_movie_path = Path(random.choice(backdrop_movie_file_paths)).as_posix()
        #
        #     road_movie_directory = os.path.join(os.path.dirname(os.path.abspath(__file__)), "road_movies")
        #     if not os.path.isdir(road_movie_directory):
        #         raise NotADirectoryError(f"{road_movie_directory} isn't a directory.")
        #     road_movie_file_paths = glob.glob(f"{road_movie_directory}/*.{movie_file_type}")
        #     if len(road_movie_file_paths) > 0:
        #         random_road_movie_path = Path(random.choice(road_movie_file_paths)).as_posix()
        # print(f"paths are\n{random_backdrop_movie_path}\n{random_road_movie_path}")

        self.holodeck_env.send_world_command(
            name="RandomizeDuckiebotsWorld",
            num_params=[float(randomize_mask)],
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
    import holodeck

    RUNname = '0606'
    if not os.path.exists('data/frame'):
        os.makedirs('data/frame')
    if not os.path.exists('data/segment'):
        os.makedirs('data/segment')

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

    image_renderer = ImageRenderer(height=60, width=80)
    image_renderer2 = ImageRenderer(height=60, width=80)

    env = UEDuckiebotsHolodeckEnv(launch_game_process=launch_game_process,
                                  render_game_on_screen=render_launched_game_on_screen,
                                  physics_hz=10)

    num_images_to_generate = 25000

    print("env initialized")
    i = 0
    while True:
        env.reset()
        env.render()
        env.render_mask()

        # These lines return the color image and mask (as BGR pixels) as np.ndarrays:
        color_image = env.render(mode="rgb_array")
        mask_image = env.render_mask(mode="rgb_array")

        episode_reward = 0.0
        episode_steps = 0
        done = False
        while not done:

            # Example to randomly select an action in the environment:
            action = env.action_space.sample()

            obs, rew, done, info = env.step(action=action)
            episode_reward += rew

            env.render()
            env.render_mask()

            # These lines return the color image and mask (as BGR pixels) as np.ndarrays:
            color_image = env.render(mode="rgb_array")
            mask_image = env.render_mask(mode="rgb_array")

            assert tuple(color_image.shape) == (60, 80, 3), color_image.shape
            assert tuple(mask_image.shape) == (60, 80, 3), mask_image.shape

            assert np.min(color_image) >= 0
            assert np.max(color_image) <= 255
            assert np.min(mask_image) >= 0
            assert np.max(mask_image) <= 255


            color_npy_save_path = f'data25k/frame/{RUNname}f{i}'
            segment_npy_save_path = f'data25k/segment/{RUNname}f{i}'

            ensure_dir(color_npy_save_path)
            ensure_dir(segment_npy_save_path)

            np.save(color_npy_save_path, color_image)
            np.save(segment_npy_save_path, mask_image)

            # rgb_data = image_renderer.render_cv2_image(cv2_image=color_image)
            # mask_data = image_renderer2.render_cv2_image(cv2_image=mask_image)
            # rgb_data = image_renderer.render_cv2_image(cv2_image=color_image)
            # mask_data = image_renderer2.render_cv2_image(cv2_image=mask_image)
            # rgb_data = image_renderer.render_cv2_image(cv2_image=color_image)
            # mask_data = image_renderer2.render_cv2_image(cv2_image=mask_image)

            color_jpeg_save_path = f'jpeg_data25k/frame/{RUNname}f{i}.jpg'
            segment_jpeg_save_path = f'jpeg_data25k/segment/{RUNname}f{i}.jpg'

            ensure_dir(color_jpeg_save_path)
            ensure_dir(segment_jpeg_save_path)

            cv2.imwrite(color_jpeg_save_path, color_image[::-1, :, :])  # save frame as JPEG file
            cv2.imwrite(segment_jpeg_save_path, mask_image[::-1, :, :])  # save frame as JPEG file

            #
            # rgb_data.save(color_jpeg_save_path)
            # mask_data.save(segment_jpeg_save_path)

            # if gamepad and gamepad.B:
            # env.reset()
            # input("press ENTER")

            episode_steps += 1
            if episode_steps > 20:
                done = True

            if i % 10 == 0:
                print(f'count: {i}')
            i += 1

            if i > num_images_to_generate:
                break




        print(f"episode reward: {episode_reward}")
        if i > num_images_to_generate:
            break