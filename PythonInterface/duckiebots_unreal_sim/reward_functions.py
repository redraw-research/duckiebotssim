import time
from abc import ABC, abstractmethod
from typing import Tuple, List
import numpy as np

class UERewardAndTerminationFunction(ABC):

    @abstractmethod
    def get_reward_and_done_for_current_timestep(self, last_action, holodeck_states_this_step: List[dict]) -> Tuple[
        float, bool]:
        pass

    @abstractmethod
    def reset(self):
        pass


class VelocityAndDistanceAlongTrackRewardAndTerminationFunctionAlwaysPenalizeTurning(UERewardAndTerminationFunction):

    # MAX_VEL_REWARD_MAGNITUDE = 1.05  # changed from 0.62 on jan 6
    MAX_VEL_REWARD_MAGNITUDE = 0.62  # changeded back on Jan 17


    MAX_ABS_FORWARD_VEL = 60.0  # changed Jan 4. Was 6 but between Oct 22 and Dec, units must have been rescaled in the game.
    #MAX_ABS_FORWARD_VEL = 6.0
    MAX_ABS_YAW_VEL = 159.0

    def __init__(self, drive_off_road_penalty_magnitude: float = 1.0):
        self._drive_off_road_penalty_magnitude = drive_off_road_penalty_magnitude

        self.reset()

    def get_reward_and_done_for_current_timestep(self, last_action, holodeck_states_this_step: List[dict]) -> Tuple[float, bool]:

        self._timesteps_elapsed += 1

        velocity_along_path = holodeck_states_this_step[-1]["ProgressAlongIntendedPathSensor"][1]
        distance_from_path = holodeck_states_this_step[-1]["ProgressAlongIntendedPathSensor"][2]
        # print(velocity_along_path)

        # print(f"distance_from_path : {holodeck_states_this_step[-1]['ProgressAlongIntendedPathSensor'][2]}")

        # reward_this_timestep = 0.0
        # reward_this_timestep -= (distance_from_path / 100.0) ** 2
        # # if distance_from_path < 70.0:
        # reward_this_timestep += np.clip(velocity_along_path, a_min=-2.0,
        #                                 a_max=self.MAX_VEL_REWARD_MAGNITUDE) / self.MAX_VEL_REWARD_MAGNITUDE
        # # else:
        # #     reward_this_timestep -= 1.0
        #
        # done = False
        # # collided_with_line = False
        # # for state in holodeck_states_this_step:
        # #     if state["DuckiebotsLineOverlapSensor"][0]:
        # #         collided_with_line = True
        # #         break
        # ## driving off-road check
        # # if collided_with_line and self._timesteps_elapsed > 1:
        # #     # Check for self._timesteps_elapsed > 1 is necessary because we may have notifications from the
        # #     # previous episode that aren't relevant to this episode.
        # #     # reward_this_timestep = -100.0
        # #     done = True


        # dreamer learns on the code below:  duckiebots_train_fill_50_000_duckiebotssim_lanefollowing_20240909_161444
        reward_this_timestep = 0.0
        # reward_this_timestep -= (distance_from_path / 100.0) ** 2
        if distance_from_path < 70.0:
            reward_this_timestep += np.clip(velocity_along_path, a_min=-2.0, a_max=self.MAX_VEL_REWARD_MAGNITUDE) / self.MAX_VEL_REWARD_MAGNITUDE
        else:
            reward_this_timestep -= 1.0

        done = False
        collided_with_line = False
        for state in holodeck_states_this_step:
            if state["DuckiebotsLineOverlapSensor"][0]:
                collided_with_line = True
                break
        # driving off-road check
        if collided_with_line and self._timesteps_elapsed > 1:
            # Check for self._timesteps_elapsed > 1 is necessary because we may have notifications from the
            # previous episode that aren't relevant to this episode.
            reward_this_timestep = -100.0
            done = True

        # # new (oct 19), encourage smooth driving
        # turn_diff = abs(current_action_input[1] - prev_action_input[1])
        # forward_diff = abs(current_action_input[0] - prev_action_input[0])
        # print(f"turn_diff: {turn_diff}, forward_diff: {forward_diff}")
        # reward_this_timestep -= (turn_diff * 0.1)
        # reward_this_timestep -= (forward_diff * 0.1)

        # # new (oct 19), encourage smooth driving stateless
        linear_forward_vel: np.ndarray = holodeck_states_this_step[-1]['DuckiebotsVelocitySensor'][0]
        yaw_vel: np.ndarray = holodeck_states_this_step[-1]['DuckiebotsVelocitySensor'][1]

        normalized_forward_vel = np.clip(linear_forward_vel,
                                        a_min=-self.MAX_ABS_FORWARD_VEL,
                                        a_max=self.MAX_ABS_FORWARD_VEL) / self.MAX_ABS_FORWARD_VEL
        normalized_rot_yaw_vel = np.clip(yaw_vel,
                                        a_min=-self.MAX_ABS_YAW_VEL,
                                        a_max=self.MAX_ABS_YAW_VEL) / self.MAX_ABS_YAW_VEL

        # 20241019_181920
        # if abs(normalized_forward_vel) > 0.3:
        #     reward_this_timestep -= 0.2 * abs(normalized_rot_yaw_vel)

        # if abs(normalized_forward_vel) > 0.2:

        # always penalize turning. Feb 21, 2025
        reward_this_timestep -= 0.5 * abs(normalized_rot_yaw_vel)

        return reward_this_timestep, done

    def reset(self):
        self._timesteps_elapsed = 0


class VelocityAndDistanceAlongTrackRewardAndTerminationFunction100Hz(UERewardAndTerminationFunction):

    # MAX_VEL_REWARD_MAGNITUDE = 1.05  # changed from 0.62 on jan 6
    MAX_VEL_REWARD_MAGNITUDE = 0.62  # changeded back on Jan 17


    MAX_ABS_FORWARD_VEL = 60.0  # changed Jan 4. Was 6 but between Oct 22 and Dec, units must have been rescaled in the game.
    #MAX_ABS_FORWARD_VEL = 6.0
    MAX_ABS_YAW_VEL = 159.0

    def __init__(self, drive_off_road_penalty_magnitude: float = 1.0):
        self._drive_off_road_penalty_magnitude = drive_off_road_penalty_magnitude

        self.reset()

    def get_reward_and_done_for_current_timestep(self, last_action, holodeck_states_this_step: List[dict]) -> Tuple[float, bool]:

        self._timesteps_elapsed += 1

        velocity_along_path = holodeck_states_this_step[-1]["ProgressAlongIntendedPathSensor"][1] * 8.333333 # 100/12
        distance_from_path = holodeck_states_this_step[-1]["ProgressAlongIntendedPathSensor"][2]
        print(velocity_along_path)

        # print(f"distance_from_path : {holodeck_states_this_step[-1]['ProgressAlongIntendedPathSensor'][2]}")

        # reward_this_timestep = 0.0
        # reward_this_timestep -= (distance_from_path / 100.0) ** 2
        # # if distance_from_path < 70.0:
        # reward_this_timestep += np.clip(velocity_along_path, a_min=-2.0,
        #                                 a_max=self.MAX_VEL_REWARD_MAGNITUDE) / self.MAX_VEL_REWARD_MAGNITUDE
        # # else:
        # #     reward_this_timestep -= 1.0
        #
        # done = False
        # # collided_with_line = False
        # # for state in holodeck_states_this_step:
        # #     if state["DuckiebotsLineOverlapSensor"][0]:
        # #         collided_with_line = True
        # #         break
        # ## driving off-road check
        # # if collided_with_line and self._timesteps_elapsed > 1:
        # #     # Check for self._timesteps_elapsed > 1 is necessary because we may have notifications from the
        # #     # previous episode that aren't relevant to this episode.
        # #     # reward_this_timestep = -100.0
        # #     done = True


        # dreamer learns on the code below:  duckiebots_train_fill_50_000_duckiebotssim_lanefollowing_20240909_161444
        reward_this_timestep = 0.0
        # reward_this_timestep -= (distance_from_path / 100.0) ** 2
        if distance_from_path < 70.0:
            reward_this_timestep += np.clip(velocity_along_path, a_min=-2.0, a_max=self.MAX_VEL_REWARD_MAGNITUDE) / self.MAX_VEL_REWARD_MAGNITUDE
        else:
            reward_this_timestep -= 1.0

        done = False
        collided_with_line = False
        for state in holodeck_states_this_step:
            if state["DuckiebotsLineOverlapSensor"][0]:
                collided_with_line = True
                break
        # driving off-road check
        if collided_with_line and self._timesteps_elapsed > 1:
            # Check for self._timesteps_elapsed > 1 is necessary because we may have notifications from the
            # previous episode that aren't relevant to this episode.
            reward_this_timestep = -100.0
            done = True

        # # new (oct 19), encourage smooth driving
        # turn_diff = abs(current_action_input[1] - prev_action_input[1])
        # forward_diff = abs(current_action_input[0] - prev_action_input[0])
        # print(f"turn_diff: {turn_diff}, forward_diff: {forward_diff}")
        # reward_this_timestep -= (turn_diff * 0.1)
        # reward_this_timestep -= (forward_diff * 0.1)

        # # new (oct 19), encourage smooth driving stateless
        linear_forward_vel: np.ndarray = holodeck_states_this_step[-1]['DuckiebotsVelocitySensor'][0]
        yaw_vel: np.ndarray = holodeck_states_this_step[-1]['DuckiebotsVelocitySensor'][1]

        normalized_forward_vel = np.clip(linear_forward_vel,
                                        a_min=-self.MAX_ABS_FORWARD_VEL,
                                        a_max=self.MAX_ABS_FORWARD_VEL) / self.MAX_ABS_FORWARD_VEL
        normalized_rot_yaw_vel = np.clip(yaw_vel,
                                        a_min=-self.MAX_ABS_YAW_VEL,
                                        a_max=self.MAX_ABS_YAW_VEL) / self.MAX_ABS_YAW_VEL

        # 20241019_181920
        # if abs(normalized_forward_vel) > 0.3:
        #     reward_this_timestep -= 0.2 * abs(normalized_rot_yaw_vel)

        # Oct 22
        if abs(normalized_forward_vel) > 0.2:
            reward_this_timestep -= 0.5 * abs(normalized_rot_yaw_vel)

        return reward_this_timestep, done

    def reset(self):
        self._timesteps_elapsed = 0



class VelocityAndDistanceAlongTrackRewardAndTerminationFunction(UERewardAndTerminationFunction):

    # MAX_VEL_REWARD_MAGNITUDE = 1.05  # changed from 0.62 on jan 6
    MAX_VEL_REWARD_MAGNITUDE = 0.62  # changeded back on Jan 17


    MAX_ABS_FORWARD_VEL = 60.0  # changed Jan 4. Was 6 but between Oct 22 and Dec, units must have been rescaled in the game.
    #MAX_ABS_FORWARD_VEL = 6.0
    MAX_ABS_YAW_VEL = 159.0

    def __init__(self, drive_off_road_penalty_magnitude: float = 1.0):
        self._drive_off_road_penalty_magnitude = drive_off_road_penalty_magnitude

        self.reset()

    def get_reward_and_done_for_current_timestep(self, last_action, holodeck_states_this_step: List[dict]) -> Tuple[float, bool]:

        self._timesteps_elapsed += 1

        velocity_along_path = holodeck_states_this_step[-1]["ProgressAlongIntendedPathSensor"][1]
        distance_from_path = holodeck_states_this_step[-1]["ProgressAlongIntendedPathSensor"][2]
        # print(velocity_along_path)

        # print(f"distance_from_path : {holodeck_states_this_step[-1]['ProgressAlongIntendedPathSensor'][2]}")

        # reward_this_timestep = 0.0
        # reward_this_timestep -= (distance_from_path / 100.0) ** 2
        # # if distance_from_path < 70.0:
        # reward_this_timestep += np.clip(velocity_along_path, a_min=-2.0,
        #                                 a_max=self.MAX_VEL_REWARD_MAGNITUDE) / self.MAX_VEL_REWARD_MAGNITUDE
        # # else:
        # #     reward_this_timestep -= 1.0
        #
        # done = False
        # # collided_with_line = False
        # # for state in holodeck_states_this_step:
        # #     if state["DuckiebotsLineOverlapSensor"][0]:
        # #         collided_with_line = True
        # #         break
        # ## driving off-road check
        # # if collided_with_line and self._timesteps_elapsed > 1:
        # #     # Check for self._timesteps_elapsed > 1 is necessary because we may have notifications from the
        # #     # previous episode that aren't relevant to this episode.
        # #     # reward_this_timestep = -100.0
        # #     done = True


        # dreamer learns on the code below:  duckiebots_train_fill_50_000_duckiebotssim_lanefollowing_20240909_161444
        reward_this_timestep = 0.0
        # reward_this_timestep -= (distance_from_path / 100.0) ** 2
        if distance_from_path < 70.0:
            reward_this_timestep += np.clip(velocity_along_path, a_min=-2.0, a_max=self.MAX_VEL_REWARD_MAGNITUDE) / self.MAX_VEL_REWARD_MAGNITUDE
        else:
            reward_this_timestep -= 1.0

        done = False
        collided_with_line = False
        for state in holodeck_states_this_step:
            if state["DuckiebotsLineOverlapSensor"][0]:
                collided_with_line = True
                break
        # driving off-road check
        if collided_with_line and self._timesteps_elapsed > 1:
            # Check for self._timesteps_elapsed > 1 is necessary because we may have notifications from the
            # previous episode that aren't relevant to this episode.
            reward_this_timestep = -100.0
            done = True

        # # new (oct 19), encourage smooth driving
        # turn_diff = abs(current_action_input[1] - prev_action_input[1])
        # forward_diff = abs(current_action_input[0] - prev_action_input[0])
        # print(f"turn_diff: {turn_diff}, forward_diff: {forward_diff}")
        # reward_this_timestep -= (turn_diff * 0.1)
        # reward_this_timestep -= (forward_diff * 0.1)

        # # new (oct 19), encourage smooth driving stateless
        linear_forward_vel: np.ndarray = holodeck_states_this_step[-1]['DuckiebotsVelocitySensor'][0]
        yaw_vel: np.ndarray = holodeck_states_this_step[-1]['DuckiebotsVelocitySensor'][1]

        normalized_forward_vel = np.clip(linear_forward_vel,
                                        a_min=-self.MAX_ABS_FORWARD_VEL,
                                        a_max=self.MAX_ABS_FORWARD_VEL) / self.MAX_ABS_FORWARD_VEL
        normalized_rot_yaw_vel = np.clip(yaw_vel,
                                        a_min=-self.MAX_ABS_YAW_VEL,
                                        a_max=self.MAX_ABS_YAW_VEL) / self.MAX_ABS_YAW_VEL

        # 20241019_181920
        # if abs(normalized_forward_vel) > 0.3:
        #     reward_this_timestep -= 0.2 * abs(normalized_rot_yaw_vel)

        # Oct 22
        if abs(normalized_forward_vel) > 0.2:
            reward_this_timestep -= 0.5 * abs(normalized_rot_yaw_vel)

        return reward_this_timestep, done

    def reset(self):
        self._timesteps_elapsed = 0

# made Jan 18
class VelocityAndDistanceAlongTrackRewardAndTerminationFunction4(UERewardAndTerminationFunction):

    MAX_VEL_REWARD_MAGNITUDE = 1.05  # changed from 0.62 on jan 6


    MAX_ABS_FORWARD_VEL = 60.0  # changed Jan 4. Was 6 but between Oct 22 and Dec, units must have been rescaled in the game.
    #MAX_ABS_FORWARD_VEL = 6.0
    MAX_ABS_YAW_VEL = 159.0

    def __init__(self, drive_off_road_penalty_magnitude: float = 1.0):
        self._drive_off_road_penalty_magnitude = drive_off_road_penalty_magnitude

        self.reset()

    def get_reward_and_done_for_current_timestep(self, last_action, holodeck_states_this_step: List[dict]) -> Tuple[float, bool]:

        self._timesteps_elapsed += 1

        velocity_along_path = holodeck_states_this_step[-1]["ProgressAlongIntendedPathSensor"][1]
        distance_from_path = holodeck_states_this_step[-1]["ProgressAlongIntendedPathSensor"][2]

        # print(f"distance_from_path : {holodeck_states_this_step[-1]['ProgressAlongIntendedPathSensor'][2]}")

        # reward_this_timestep = 0.0
        # reward_this_timestep -= (distance_from_path / 100.0) ** 2
        # # if distance_from_path < 70.0:
        # reward_this_timestep += np.clip(velocity_along_path, a_min=-2.0,
        #                                 a_max=self.MAX_VEL_REWARD_MAGNITUDE) / self.MAX_VEL_REWARD_MAGNITUDE
        # # else:
        # #     reward_this_timestep -= 1.0
        #
        # done = False
        # # collided_with_line = False
        # # for state in holodeck_states_this_step:
        # #     if state["DuckiebotsLineOverlapSensor"][0]:
        # #         collided_with_line = True
        # #         break
        # ## driving off-road check
        # # if collided_with_line and self._timesteps_elapsed > 1:
        # #     # Check for self._timesteps_elapsed > 1 is necessary because we may have notifications from the
        # #     # previous episode that aren't relevant to this episode.
        # #     # reward_this_timestep = -100.0
        # #     done = True


        # dreamer learns on the code below:  duckiebots_train_fill_50_000_duckiebotssim_lanefollowing_20240909_161444
        reward_this_timestep = 0.0
        # reward_this_timestep -= (distance_from_path / 100.0) ** 2
        if distance_from_path < 60.0:
            reward_this_timestep += np.clip(velocity_along_path, a_min=-2.0, a_max=self.MAX_VEL_REWARD_MAGNITUDE) / self.MAX_VEL_REWARD_MAGNITUDE
        else:
            reward_this_timestep -= 1.0

        done = False
        collided_with_line = False
        for state in holodeck_states_this_step:
            if state["DuckiebotsLineOverlapSensor"][0]:
                collided_with_line = True
                break
        # driving off-road check
        if collided_with_line and self._timesteps_elapsed > 1:
            # Check for self._timesteps_elapsed > 1 is necessary because we may have notifications from the
            # previous episode that aren't relevant to this episode.
            reward_this_timestep = -100.0
            done = True

        # # new (oct 19), encourage smooth driving
        # turn_diff = abs(current_action_input[1] - prev_action_input[1])
        # forward_diff = abs(current_action_input[0] - prev_action_input[0])
        # print(f"turn_diff: {turn_diff}, forward_diff: {forward_diff}")
        # reward_this_timestep -= (turn_diff * 0.1)
        # reward_this_timestep -= (forward_diff * 0.1)

        # # new (oct 19), encourage smooth driving stateless
        linear_forward_vel: np.ndarray = holodeck_states_this_step[-1]['DuckiebotsVelocitySensor'][0]
        yaw_vel: np.ndarray = holodeck_states_this_step[-1]['DuckiebotsVelocitySensor'][1]

        normalized_forward_vel = np.clip(linear_forward_vel,
                                        a_min=-self.MAX_ABS_FORWARD_VEL,
                                        a_max=self.MAX_ABS_FORWARD_VEL) / self.MAX_ABS_FORWARD_VEL
        normalized_rot_yaw_vel = np.clip(yaw_vel,
                                        a_min=-self.MAX_ABS_YAW_VEL,
                                        a_max=self.MAX_ABS_YAW_VEL) / self.MAX_ABS_YAW_VEL

        # 20241019_181920
        # if abs(normalized_forward_vel) > 0.3:
        #     reward_this_timestep -= 0.2 * abs(normalized_rot_yaw_vel)

        # Oct 22
        if abs(normalized_forward_vel) > 0.2:
            reward_this_timestep -= 1.0 * abs(normalized_rot_yaw_vel) # changed from 0.5 Jan 18

        return reward_this_timestep, done

    def reset(self):
        self._timesteps_elapsed = 0


# January 2, to reproduce 20241019_181920
class VelocityAndDistanceAlongTrackRewardAndTerminationFunction2(UERewardAndTerminationFunction):

    MAX_VEL_REWARD_MAGNITUDE = 1.05  # changed from 0.62 on jan 6

    MAX_ABS_FORWARD_VEL = 60.0  # changed Jan 4. Was 6 but between Oct 22 and Dec, units must have been rescaled in the game.
    #MAX_ABS_FORWARD_VEL = 6.0
    MAX_ABS_YAW_VEL = 159.0

    def __init__(self, drive_off_road_penalty_magnitude: float = 1.0):
        self._drive_off_road_penalty_magnitude = drive_off_road_penalty_magnitude

        self.reset()

    def get_reward_and_done_for_current_timestep(self, last_action, holodeck_states_this_step: List[dict]) -> Tuple[float, bool]:

        self._timesteps_elapsed += 1

        velocity_along_path = holodeck_states_this_step[-1]["ProgressAlongIntendedPathSensor"][1]
        distance_from_path = holodeck_states_this_step[-1]["ProgressAlongIntendedPathSensor"][2]

        # reward_this_timestep = 0.0
        # reward_this_timestep -= (distance_from_path / 100.0) ** 2
        # # if distance_from_path < 70.0:
        # reward_this_timestep += np.clip(velocity_along_path, a_min=-2.0,
        #                                 a_max=self.MAX_VEL_REWARD_MAGNITUDE) / self.MAX_VEL_REWARD_MAGNITUDE
        # # else:
        # #     reward_this_timestep -= 1.0
        #
        # done = False
        # # collided_with_line = False
        # # for state in holodeck_states_this_step:
        # #     if state["DuckiebotsLineOverlapSensor"][0]:
        # #         collided_with_line = True
        # #         break
        # ## driving off-road check
        # # if collided_with_line and self._timesteps_elapsed > 1:
        # #     # Check for self._timesteps_elapsed > 1 is necessary because we may have notifications from the
        # #     # previous episode that aren't relevant to this episode.
        # #     # reward_this_timestep = -100.0
        # #     done = True


        # dreamer learns on the code below:  duckiebots_train_fill_50_000_duckiebotssim_lanefollowing_20240909_161444
        reward_this_timestep = 0.0
        # reward_this_timestep -= (distance_from_path / 100.0) ** 2
        if distance_from_path < 70.0:
            reward_this_timestep += np.clip(velocity_along_path, a_min=-2.0, a_max=self.MAX_VEL_REWARD_MAGNITUDE) / self.MAX_VEL_REWARD_MAGNITUDE
        else:
            reward_this_timestep -= 1.0

        done = False
        collided_with_line = False
        for state in holodeck_states_this_step:
            if state["DuckiebotsLineOverlapSensor"][0]:
                collided_with_line = True
                break
        # driving off-road check
        if collided_with_line and self._timesteps_elapsed > 1:
            # Check for self._timesteps_elapsed > 1 is necessary because we may have notifications from the
            # previous episode that aren't relevant to this episode.
            reward_this_timestep = -100.0
            done = True

        # # new (oct 19), encourage smooth driving
        # turn_diff = abs(current_action_input[1] - prev_action_input[1])
        # forward_diff = abs(current_action_input[0] - prev_action_input[0])
        # print(f"turn_diff: {turn_diff}, forward_diff: {forward_diff}")
        # reward_this_timestep -= (turn_diff * 0.1)
        # reward_this_timestep -= (forward_diff * 0.1)

        # # new (oct 19), encourage smooth driving stateless
        linear_forward_vel: np.ndarray = holodeck_states_this_step[-1]['DuckiebotsVelocitySensor'][0]
        # print(f"raw linear forward vel: {linear_forward_vel}")
        yaw_vel: np.ndarray = holodeck_states_this_step[-1]['DuckiebotsVelocitySensor'][1]

        normalized_forward_vel = np.clip(linear_forward_vel,
                                        a_min=-self.MAX_ABS_FORWARD_VEL,
                                        a_max=self.MAX_ABS_FORWARD_VEL) / self.MAX_ABS_FORWARD_VEL
        normalized_rot_yaw_vel = np.clip(yaw_vel,
                                        a_min=-self.MAX_ABS_YAW_VEL,
                                        a_max=self.MAX_ABS_YAW_VEL) / self.MAX_ABS_YAW_VEL

        # 20241019_181920
        if abs(normalized_forward_vel) > 0.3:
            reward_this_timestep -= 0.2 * abs(normalized_rot_yaw_vel)

        # Oct 22
        # if abs(normalized_forward_vel) > 0.2:
        #     reward_this_timestep -= 0.5 * abs(normalized_rot_yaw_vel)

        return reward_this_timestep, done

    def reset(self):
        self._timesteps_elapsed = 0


class VelocityAndDistanceAlongTrackRewardAndTerminationFunction3(UERewardAndTerminationFunction):

    MAX_VEL_REWARD_MAGNITUDE = 1.05  # changed from 0.62 on jan 6

    MAX_ABS_FORWARD_VEL = 60.0  # changed Jan 4. Was 6 but between Oct 22 and Dec, units must have been rescaled in the game.
    #MAX_ABS_FORWARD_VEL = 6.0
    MAX_ABS_YAW_VEL = 159.0

    def __init__(self, drive_off_road_penalty_magnitude: float = 1.0):
        self._drive_off_road_penalty_magnitude = drive_off_road_penalty_magnitude

        self.reset()

    def get_reward_and_done_for_current_timestep(self, last_action, holodeck_states_this_step: List[dict]) -> Tuple[float, bool]:

        self._timesteps_elapsed += 1

        velocity_along_path = holodeck_states_this_step[-1]["ProgressAlongIntendedPathSensor"][1]
        distance_from_path = holodeck_states_this_step[-1]["ProgressAlongIntendedPathSensor"][2]
        # reward_this_timestep = 0.0
        # reward_this_timestep -= (distance_from_path / 100.0) ** 2
        # # if distance_from_path < 70.0:
        # reward_this_timestep += np.clip(velocity_along_path, a_min=-2.0,
        #                                 a_max=self.MAX_VEL_REWARD_MAGNITUDE) / self.MAX_VEL_REWARD_MAGNITUDE
        # # else:
        # #     reward_this_timestep -= 1.0
        #
        # done = False
        # # collided_with_line = False
        # # for state in holodeck_states_this_step:
        # #     if state["DuckiebotsLineOverlapSensor"][0]:
        # #         collided_with_line = True
        # #         break
        # ## driving off-road check
        # # if collided_with_line and self._timesteps_elapsed > 1:
        # #     # Check for self._timesteps_elapsed > 1 is necessary because we may have notifications from the
        # #     # previous episode that aren't relevant to this episode.
        # #     # reward_this_timestep = -100.0
        # #     done = True


        # dreamer learns on the code below:  duckiebots_train_fill_50_000_duckiebotssim_lanefollowing_20240909_161444
        reward_this_timestep = 0.0
        # reward_this_timestep -= (distance_from_path / 100.0) ** 2
        if distance_from_path < 70.0:
            reward_this_timestep += np.clip(velocity_along_path, a_min=-2.0, a_max=self.MAX_VEL_REWARD_MAGNITUDE) / self.MAX_VEL_REWARD_MAGNITUDE
        else:
            reward_this_timestep -= 1.0

        done = False
        collided_with_line = False
        for state in holodeck_states_this_step:
            if state["DuckiebotsLineOverlapSensor"][0]:
                collided_with_line = True
                break
        # driving off-road check
        if collided_with_line and self._timesteps_elapsed > 1:
            # Check for self._timesteps_elapsed > 1 is necessary because we may have notifications from the
            # previous episode that aren't relevant to this episode.
            reward_this_timestep = -100.0
            done = True

        # # new (oct 19), encourage smooth driving
        # turn_diff = abs(current_action_input[1] - prev_action_input[1])
        # forward_diff = abs(current_action_input[0] - prev_action_input[0])
        # print(f"turn_diff: {turn_diff}, forward_diff: {forward_diff}")
        # reward_this_timestep -= (turn_diff * 0.1)
        # reward_this_timestep -= (forward_diff * 0.1)

        # # new (oct 19), encourage smooth driving stateless
        linear_forward_vel: np.ndarray = holodeck_states_this_step[-1]['DuckiebotsVelocitySensor'][0]
        yaw_vel: np.ndarray = holodeck_states_this_step[-1]['DuckiebotsVelocitySensor'][1]

        normalized_forward_vel = np.clip(linear_forward_vel,
                                        a_min=-self.MAX_ABS_FORWARD_VEL,
                                        a_max=self.MAX_ABS_FORWARD_VEL) / self.MAX_ABS_FORWARD_VEL
        normalized_rot_yaw_vel = np.clip(yaw_vel,
                                        a_min=-self.MAX_ABS_YAW_VEL,
                                        a_max=self.MAX_ABS_YAW_VEL) / self.MAX_ABS_YAW_VEL

        # 20241019_181920
        # if abs(normalized_forward_vel) > 0.3:
        #     reward_this_timestep -= 0.2 * abs(normalized_rot_yaw_vel)

        # Oct 22
        # if abs(normalized_forward_vel) > 0.2:
        #     reward_this_timestep -= 0.5 * abs(normalized_rot_yaw_vel)

        return reward_this_timestep, done

    def reset(self):
        self._timesteps_elapsed = 0


class VelocityAndDistanceAlongTrackRewardAndTerminationFunctionStrictDistance(UERewardAndTerminationFunction):

    MAX_VEL_REWARD_MAGNITUDE = 1.05  # changed from 0.62 on jan 6

    MAX_ABS_FORWARD_VEL = 60.0  # changed Jan 4. Was 6 but between Oct 22 and Dec, units must have been rescaled in the game.
    #MAX_ABS_FORWARD_VEL = 6.0
    MAX_ABS_YAW_VEL = 159.0

    def __init__(self, drive_off_road_penalty_magnitude: float = 1.0):
        self._drive_off_road_penalty_magnitude = drive_off_road_penalty_magnitude

        self.reset()

    def get_reward_and_done_for_current_timestep(self, last_action, holodeck_states_this_step: List[dict]) -> Tuple[float, bool]:

        self._timesteps_elapsed += 1

        velocity_along_path = holodeck_states_this_step[-1]["ProgressAlongIntendedPathSensor"][1]
        distance_from_path = holodeck_states_this_step[-1]["ProgressAlongIntendedPathSensor"][2]

        # reward_this_timestep = 0.0
        # reward_this_timestep -= (distance_from_path / 100.0) ** 2
        # # if distance_from_path < 70.0:
        # reward_this_timestep += np.clip(velocity_along_path, a_min=-2.0,
        #                                 a_max=self.MAX_VEL_REWARD_MAGNITUDE) / self.MAX_VEL_REWARD_MAGNITUDE
        # # else:
        # #     reward_this_timestep -= 1.0
        #
        # done = False
        # # collided_with_line = False
        # # for state in holodeck_states_this_step:
        # #     if state["DuckiebotsLineOverlapSensor"][0]:
        # #         collided_with_line = True
        # #         break
        # ## driving off-road check
        # # if collided_with_line and self._timesteps_elapsed > 1:
        # #     # Check for self._timesteps_elapsed > 1 is necessary because we may have notifications from the
        # #     # previous episode that aren't relevant to this episode.
        # #     # reward_this_timestep = -100.0
        # #     done = True


        # dreamer learns on the code below:  duckiebots_train_fill_50_000_duckiebotssim_lanefollowing_20240909_161444
        reward_this_timestep = 0.0
        # reward_this_timestep -= (distance_from_path / 100.0) ** 2
        if distance_from_path < 50.0:
            reward_this_timestep += np.clip(velocity_along_path, a_min=-2.0, a_max=self.MAX_VEL_REWARD_MAGNITUDE) / self.MAX_VEL_REWARD_MAGNITUDE
        else:
            reward_this_timestep -= 1.0

        done = False
        collided_with_line = False
        for state in holodeck_states_this_step:
            if state["DuckiebotsLineOverlapSensor"][0]:
                collided_with_line = True
                break
        # driving off-road check
        if collided_with_line and self._timesteps_elapsed > 1:
            # Check for self._timesteps_elapsed > 1 is necessary because we may have notifications from the
            # previous episode that aren't relevant to this episode.
            reward_this_timestep = -100.0
            done = True

        # # new (oct 19), encourage smooth driving
        # turn_diff = abs(current_action_input[1] - prev_action_input[1])
        # forward_diff = abs(current_action_input[0] - prev_action_input[0])
        # print(f"turn_diff: {turn_diff}, forward_diff: {forward_diff}")
        # reward_this_timestep -= (turn_diff * 0.1)
        # reward_this_timestep -= (forward_diff * 0.1)

        # # new (oct 19), encourage smooth driving stateless
        linear_forward_vel: np.ndarray = holodeck_states_this_step[-1]['DuckiebotsVelocitySensor'][0]
        yaw_vel: np.ndarray = holodeck_states_this_step[-1]['DuckiebotsVelocitySensor'][1]

        normalized_forward_vel = np.clip(linear_forward_vel,
                                        a_min=-self.MAX_ABS_FORWARD_VEL,
                                        a_max=self.MAX_ABS_FORWARD_VEL) / self.MAX_ABS_FORWARD_VEL
        normalized_rot_yaw_vel = np.clip(yaw_vel,
                                        a_min=-self.MAX_ABS_YAW_VEL,
                                        a_max=self.MAX_ABS_YAW_VEL) / self.MAX_ABS_YAW_VEL

        # 20241019_181920
        # if abs(normalized_forward_vel) > 0.3:
        #     reward_this_timestep -= 0.2 * abs(normalized_rot_yaw_vel)

        # Oct 22
        if abs(normalized_forward_vel) > 0.2:
            reward_this_timestep -= 0.5 * abs(normalized_rot_yaw_vel)

        return reward_this_timestep, done

    def reset(self):
        self._timesteps_elapsed = 0


class VelocityAlongTrackRewardAndTerminationFunction(UERewardAndTerminationFunction):

    MAX_VEL_REWARD_MAGNITUDE = 1.05  # changed from 0.62 on jan 6

    def __init__(self, drive_off_road_penalty_magnitude: float = 1.0):
        self._drive_off_road_penalty_magnitude = drive_off_road_penalty_magnitude

        self.reset()

    def get_reward_and_done_for_current_timestep(self, last_action, holodeck_states_this_step: List[dict]) -> Tuple[float, bool]:

        self._timesteps_elapsed += 1

        reward_this_timestep = holodeck_states_this_step[-1]["ProgressAlongIntendedPathSensor"][1]

        # hack to address rare cases when velocity along track is larger than the expected max velocity
        reward_this_timestep = np.clip(reward_this_timestep, a_min=-2.0, a_max=self.MAX_VEL_REWARD_MAGNITUDE) / self.MAX_VEL_REWARD_MAGNITUDE

        done = False

        collided_with_line = False
        for state in holodeck_states_this_step:
            if state["DuckiebotsLineOverlapSensor"][0]:
                collided_with_line = True
                break

        # driving off-road check
        if collided_with_line and self._timesteps_elapsed > 1:
            # Check for self._timesteps_elapsed > 1 is necessary because we may have notifications from the
            # previous episode that aren't relevant to this episode.
            reward_this_timestep -= self._drive_off_road_penalty_magnitude
            # done = True

        return reward_this_timestep, done

    def reset(self):
        self._timesteps_elapsed = 0

class DefaultDuckiebotsRewardAndTerminationFunction(UERewardAndTerminationFunction):

    def __init__(self,
                 max_episode_length: int = 10000,
                 drive_off_road_penalty_magnitude: float = 1.0,
                 wrong_way_penalty_magnitude: float = 0.1,
                 visit_new_tile_reward: float = 10.0):

        self._max_episode_length = max_episode_length
        self._drive_off_road_penalty_magnitude = drive_off_road_penalty_magnitude
        self._wrong_way_penalty_magnitude = wrong_way_penalty_magnitude
        self._visit_new_tile_reward = visit_new_tile_reward

        self.reset()

    def get_reward_and_done_for_current_timestep(self, last_action, holodeck_states_this_step: List[dict]) -> Tuple[
        float, bool]:

        self._timesteps_elapsed += 1

        # Special case where we may have notifications from the previous episode that aren't relevant to this episode.
        if self._timesteps_elapsed == 1:
            return 0.0, False

        done = False
        reward_this_timestep = 0.0

        # max episode length check
        if self._timesteps_elapsed > self._max_episode_length:
            # print("hit max timesteps")
            done = True

        collided_with_white_line = False
        collided_with_yellow_line = False
        is_yellow_line_to_left_and_white_line_to_right = True
        entered_new_road_tile = False

        for state in holodeck_states_this_step:
            # CollidedWithWhiteLine, CollidedWithYellowLine, IsYellowLineToLeftAndWhiteLineToRight, EnteredNewRoadTile
            (state_collided_with_white_line,
             state_collided_with_yellow_line,
             state_is_yellow_line_to_left_and_white_line_to_right,
             state_entered_new_road_tile) = state["DuckiebotsLoopStatusSensor"]

            # tracking if an event ever occurred
            if state_collided_with_white_line:
                collided_with_white_line = True
            if state_collided_with_yellow_line:
                collided_with_yellow_line = True
            if state_entered_new_road_tile:
                entered_new_road_tile = True

            # tracking the latest status
            is_yellow_line_to_left_and_white_line_to_right = state_is_yellow_line_to_left_and_white_line_to_right

        # driving off-road check
        if (collided_with_white_line or collided_with_yellow_line) and self._timesteps_elapsed > 1:
            reward_this_timestep -= self._drive_off_road_penalty_magnitude
            # print("collided with line")
            done = True

        # driving wrong way check
        if not is_yellow_line_to_left_and_white_line_to_right:
            reward_this_timestep -= self._wrong_way_penalty_magnitude

        # visiting new tiles check
        if entered_new_road_tile:
            reward_this_timestep += self._visit_new_tile_reward

        return reward_this_timestep, done

    def reset(self):
        # tracked stats:
        self._timesteps_elapsed = 0


class DenseDuckiebotsRewardAndTerminationFunction(UERewardAndTerminationFunction):

    def __init__(self,
                 max_episode_length: int = 10000,
                 drive_off_road_penalty_magnitude: float = 1.0,
                 wrong_way_penalty_magnitude: float = 0.1,
                 visit_new_tile_reward: float = 1.0):

        self._max_episode_length = max_episode_length
        self._drive_off_road_penalty_magnitude = drive_off_road_penalty_magnitude
        self._wrong_way_penalty_magnitude = wrong_way_penalty_magnitude
        self._visit_new_tile_reward = visit_new_tile_reward
        self.reset()

    def get_reward_and_done_for_current_timestep(self, current_action_input, holodeck_states_this_step: List[dict]) -> Tuple[
        float, bool]:

        latest_action_velocity, latest_action_turning = current_action_input

        self._timesteps_elapsed += 1

        # Special case where we may have notifications from the previous episode that aren't relevant to this episode.
        if self._timesteps_elapsed == 1:
            return 0.0, False

        done = False
        reward_this_timestep = 0.0

        # max episode length check
        if self._timesteps_elapsed > self._max_episode_length:
            # print("hit max timesteps")
            done = True

        collided_with_white_line = False
        collided_with_yellow_line = False
        is_yellow_line_to_left_and_white_line_to_right = True
        entered_new_road_tile = False

        for state in holodeck_states_this_step:
            # CollidedWithWhiteLine, CollidedWithYellowLine, IsYellowLineToLeftAndWhiteLineToRight, EnteredNewRoadTile
            (state_collided_with_white_line,
             state_collided_with_yellow_line,
             state_is_yellow_line_to_left_and_white_line_to_right,
             state_entered_new_road_tile) = state["DuckiebotsLoopStatusSensor"]

            # tracking if an event ever occurred
            if state_collided_with_white_line:
                collided_with_white_line = True
            if state_collided_with_yellow_line:
                collided_with_yellow_line = True
            if state_entered_new_road_tile:
                entered_new_road_tile = True

            # tracking the latest status
            is_yellow_line_to_left_and_white_line_to_right = state_is_yellow_line_to_left_and_white_line_to_right

        # driving off-road check
        if (collided_with_white_line or collided_with_yellow_line) and self._timesteps_elapsed > 1:
            reward_this_timestep -= self._drive_off_road_penalty_magnitude
            # print("collided with line")
            done = True

        # driving wrong way check
        if not is_yellow_line_to_left_and_white_line_to_right:
            reward_this_timestep -= self._wrong_way_penalty_magnitude
        # elif latest_action_velocity > 0.6:
        #     # reward hack to go forward
        #     reward_this_timestep += 0.01 * self._wrong_way_penalty_magnitude
        #
        # # reward hack to minimize turns
        # turn_amount = last_action[1]
        # reward_this_timestep -= 0.2 * abs(turn_amount) * self._wrong_way_penalty_magnitude

        # visiting new tiles check
        if entered_new_road_tile:
            reward_this_timestep += self._visit_new_tile_reward

        return reward_this_timestep, done

    def reset(self):
        # tracked stats:
        self._timesteps_elapsed = 0



class DenseDuckiebotsRewardAndTerminationFunction2(UERewardAndTerminationFunction):

    def __init__(self,
                 max_episode_length: int = 10000,
                 drive_off_road_penalty_magnitude: float = 10.0,
                 wrong_way_penalty_magnitude: float = 0.1,
                 visit_new_tile_reward: float = 5.0):

        self._max_episode_length = max_episode_length
        self._drive_off_road_penalty_magnitude = drive_off_road_penalty_magnitude
        self._wrong_way_penalty_magnitude = wrong_way_penalty_magnitude
        self._visit_new_tile_reward = visit_new_tile_reward
        self.reset()

    def get_reward_and_done_for_current_timestep(self, current_action_input, holodeck_states_this_step: List[dict]) -> Tuple[
        float, bool]:

        latest_action_velocity, latest_action_turning = current_action_input

        self._timesteps_elapsed += 1

        # Special case where we may have notifications from the previous episode that aren't relevant to this episode.
        if self._timesteps_elapsed == 1:
            return 0.0, False

        done = False
        reward_this_timestep = 0.0

        # max episode length check
        if self._timesteps_elapsed > self._max_episode_length:
            # print("hit max timesteps")
            done = True

        collided_with_white_line = False
        collided_with_yellow_line = False
        is_yellow_line_to_left_and_white_line_to_right = True
        entered_new_road_tile = False

        for state in holodeck_states_this_step:
            # CollidedWithWhiteLine, CollidedWithYellowLine, IsYellowLineToLeftAndWhiteLineToRight, EnteredNewRoadTile
            (state_collided_with_white_line,
             state_collided_with_yellow_line,
             state_is_yellow_line_to_left_and_white_line_to_right,
             state_entered_new_road_tile) = state["DuckiebotsLoopStatusSensor"]

            # tracking if an event ever occurred
            if state_collided_with_white_line:
                collided_with_white_line = True
            if state_collided_with_yellow_line:
                collided_with_yellow_line = True
            if state_entered_new_road_tile:
                entered_new_road_tile = True

            # tracking the latest status
            is_yellow_line_to_left_and_white_line_to_right = state_is_yellow_line_to_left_and_white_line_to_right

        # driving off-road check
        if (collided_with_white_line or collided_with_yellow_line) and self._timesteps_elapsed > 1:
            reward_this_timestep -= self._drive_off_road_penalty_magnitude
            # print("collided with line")
            done = True

        # driving wrong way check
        if not is_yellow_line_to_left_and_white_line_to_right:
            reward_this_timestep -= self._wrong_way_penalty_magnitude
        elif latest_action_velocity > 0.6:
            # reward hack to go forward
            reward_this_timestep += 0.4 * self._wrong_way_penalty_magnitude

        # # reward hack to minimize turns
        # turn_amount = last_action[1]
        # reward_this_timestep -= 0.05 * abs(turn_amount) * self._wrong_way_penalty_magnitude

        # visiting new tiles check
        if entered_new_road_tile:
            reward_this_timestep += self._visit_new_tile_reward

        return reward_this_timestep, done

    def reset(self):
        # tracked stats:
        self._timesteps_elapsed = 0
