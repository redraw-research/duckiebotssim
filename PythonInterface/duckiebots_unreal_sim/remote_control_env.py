import time
from typing import Optional, Tuple

import cv2
import gym
import numpy as np
from gym.spaces import Box

from duckiebots_unreal_sim.level_randomization import randomize_level
from duckiebots_unreal_sim.reward_functions import UERewardAndTerminationFunction, \
    DefaultDuckiebotsRewardAndTerminationFunction
from duckiebots_unreal_sim.tools import UEHTTPBridge, UEProcessManager, ImageRenderer, NDIImageReceiver, Rate, \
    DEFAULT_GAME_LAUNCHER_PATH


def _action_ctrl_json(forward_vel: float, turn_amount: float):
    return {"ForwardVel": -float(forward_vel), "TurnAmount": float(turn_amount)}


class UEDuckiebotsRemoteControlEnv(gym.Env):

    def __init__(self,
                 target_control_hz: float = 10.,
                 observation_out_height: int = 84,
                 observation_out_width: int = 84,
                 randomization_enabled: bool = True,
                 randomize_movie: bool = True,
                 reward_termination_function: Optional[UERewardAndTerminationFunction] = None,
                 launch_game_process: bool = False,
                 game_launcher_path: str = DEFAULT_GAME_LAUNCHER_PATH,
                 render_launched_game_process_offscreen: bool = False):

        self._randomize_movie = randomize_movie
        self._randomization_enabled = randomization_enabled

        self._action_rate = Rate(target_hz=target_control_hz)
        self._observation_out_height = observation_out_height
        self._observation_out_width = observation_out_width

        self._renderer = None
        self._latest_obs_out_rgb_hwc_obs = None

        self._action_publish_thread = None
        self._last_action_ctrl_json = _action_ctrl_json(forward_vel=0.0, turn_amount=0.0)
        self._time_of_last_action = 0.0

        self._shutdown = False
        self._has_finished_shutdown = False

        self.action_space = Box(low=-1.0, high=1.0, shape=(2,), dtype=np.float32)
        self.observation_space = Box(low=0,
                                     high=256,
                                     shape=(observation_out_height, observation_out_width, 3),
                                     dtype=np.float32)

        self._reward_termination_function = reward_termination_function
        if self._reward_termination_function is None:
            self._reward_termination_function = DefaultDuckiebotsRewardAndTerminationFunction()

        self._process_manager = None
        if launch_game_process:
            self._process_manager = UEProcessManager(game_launcher_file_path=game_launcher_path)
            self._process_manager.launch_game(render_off_screen=render_launched_game_process_offscreen)
            time.sleep(10)

        self._http_bridge = UEHTTPBridge()
        self._image_receiver = NDIImageReceiver()

        self._wait_for_initial_first_camera_frame()

    def _wait_for_initial_first_camera_frame(self):
        rate = Rate(target_hz=5)
        i = 0
        while self._image_receiver.get_latest_image() is None:
            if i % 25 == 0:
                print(f"Waiting for initial first image")
            rate.sleep()
            i += 1

    def _format_obs(self, bgr_hwc_obs: np.ndarray):
        # Resize image
        bgr_hwc_obs = cv2.resize(bgr_hwc_obs,
                                 (self._observation_out_width, self._observation_out_height),
                                 interpolation=cv2.INTER_AREA)
        # BGR to RGB
        rgb_hwc_obs = bgr_hwc_obs[:, :, ::-1]
        # # # Reorder Axes (Height x Width x Channel) -> (Channel x Height x Width)
        # rgb_chw_obs = np.transpose(rgb_hwc_obs, axes=(2, 0, 1))
        return rgb_hwc_obs

    def reset(self, seed: Optional[int] = None, options: Optional[dict] = None) -> np.ndarray:
        if self._randomization_enabled:
            self.randomize_level(randomize_movie=self._randomize_movie)
        self._http_bridge.make_ue_function_call(function_name="MoveAgentToRandomLocation")
        self._reward_termination_function.reset()
        time.sleep(0.5)  # (author1) Need a better way to wait for a new video backdrop to load in Unreal, etc.
        bgr_hwc_obs = self._image_receiver.get_latest_image()[::-1, :, :3]
        rgb_hwc_obs = self._format_obs(bgr_hwc_obs=bgr_hwc_obs)
        self._latest_obs_out_rgb_hwc_obs = rgb_hwc_obs
        assert rgb_hwc_obs in self.observation_space, (rgb_hwc_obs.shape, rgb_hwc_obs.max(), rgb_hwc_obs.min())
        self._last_action_ctrl_json = _action_ctrl_json(forward_vel=0.0, turn_amount=0.0)
        return rgb_hwc_obs

    def process_action(self, action: np.ndarray) -> np.ndarray:
        # Override this method in a subclass to implement better handling.
        deadzone = 0.15
        action[np.abs(action) < deadzone] = 0.0
        return action

    def step(self, action: np.ndarray) -> Tuple[np.ndarray, float, bool, dict]:
        if action not in self.action_space:
            raise ValueError(f"Action {action} is not in the action space {self.action_space}.")
        action = self.process_action(action=action)
        assert action in self.action_space, action

        self._last_action_ctrl_json = _action_ctrl_json(forward_vel=action[0], turn_amount=action[1])
        self._time_of_last_action = time.time()

        self._action_rate.sleep()

        json_notifications = self._http_bridge.make_ue_function_call(
            function_name="ApplyRemoteControlMovementInput",
            function_parameters=self._last_action_ctrl_json)

        rew, done = self._reward_termination_function.get_reward_and_done_for_current_timestep(
            latest_unreal_engine_json_notification=json_notifications
        )

        bgr_hwc_obs = self._image_receiver.get_latest_image()[::-1, :, :3]
        rgb_hwc_obs = self._format_obs(bgr_hwc_obs=bgr_hwc_obs)
        self._latest_obs_out_rgb_hwc_obs = rgb_hwc_obs

        return rgb_hwc_obs, rew, done, json_notifications

    def render(self, mode="human") -> Optional[np.ndarray]:
        image_to_render_bgr_hwc = np.copy(self._latest_obs_out_rgb_hwc_obs[:, :, ::-1])
        if mode == "human":
            if not self._renderer:
                height = image_to_render_bgr_hwc.shape[0]
                width = image_to_render_bgr_hwc.shape[1]
                self._renderer = ImageRenderer(height=height, width=width)
            if image_to_render_bgr_hwc is not None:
                self._renderer.render_cv2_image(image_to_render_bgr_hwc)
        elif mode == "rgb_array":
            return image_to_render_bgr_hwc
        else:
            raise ValueError(f"Unknown render mode: {mode}")

    def close(self):
        self._shutdown = True
        if not self._has_finished_shutdown:
            print("Shutting Down Environment")
            if self._process_manager:
                self._process_manager.close()
            if self._image_receiver:
                self._image_receiver.close()
            if self._renderer:
                self._renderer.close()
            if self._action_publish_thread:
                self._action_publish_thread.join(timeout=0.5)
            self._has_finished_shutdown = True

    def report_avg_control_hz(self) -> float:
        return self._action_rate.report_avg_hz()

    def clear_control_hz_metrics(self):
        self._action_rate.clear_metrics()

    def randomize_level(self, randomize_movie: bool = True):
        randomize_level(http_bridge=self._http_bridge, randomize_movie=randomize_movie)

    def __del__(self):
        self.close()

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.close()


if __name__ == '__main__':
    from tools import XboxController, get_keyboard_turning, get_keyboard_velocity

    print("Detecting gamepad (you may have to press a button on the controller)...")
    gamepad = None
    if XboxController.detect_gamepad():
        gamepad = XboxController()
    print("Gamepad found" if gamepad else "use keyboard controls")

    launch_game_process = False
    render_launched_game_offscreen = False
    game_launcher_path = DEFAULT_GAME_LAUNCHER_PATH

    with UEDuckiebotsRemoteControlEnv(launch_game_process=launch_game_process,
                                      game_launcher_path=DEFAULT_GAME_LAUNCHER_PATH,
                                      render_launched_game_process_offscreen=render_launched_game_offscreen) as env:
        print("env initialized")
        while True:
            i = 0
            env.reset()
            env.render()
            done = False
            while not done:
                velocity = gamepad.LeftJoystickY if gamepad else get_keyboard_velocity()
                turning = gamepad.LeftJoystickX if gamepad else get_keyboard_turning()
                action = np.asarray([velocity, turning], np.float32)
                obs, rew, done, info = env.step(action=action)
                env.render()
                if i % 100 == 0:
                    print(f"avg control hz: {env.report_avg_control_hz()}")
                i += 1
