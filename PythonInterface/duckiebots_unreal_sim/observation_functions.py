from abc import ABC, abstractmethod

import cv2
import numpy as np


class UEObservationFunction(ABC):

    @abstractmethod
    def get_observation_for_timestep(self, holodeck_state: dict):
        raise NotImplementedError


class UEDuckiebotsRGBCameraObservationFunction(UEObservationFunction):

    def __init__(self, obs_out_height, obs_out_width):
        self._obs_out_height = obs_out_height
        self._obs_out_width = obs_out_width

    def get_observation_for_timestep(self, holodeck_state: dict) -> np.ndarray:
        bgr_hwc_color_image: np.ndarray = holodeck_state["RGBCamera"][:, :, :3]
        rgb_hwc_color_image = bgr_hwc_color_image[:, :, ::-1]

        rgb_hwc_color_image = cv2.resize(rgb_hwc_color_image,
                                         (self._obs_out_width, self._obs_out_height),
                                         interpolation=cv2.INTER_AREA).astype(dtype=np.uint8)

        return rgb_hwc_color_image


class UEDuckiebotsSemanticMaskCameraObservationFunction(UEObservationFunction):

    def __init__(self, obs_out_height, obs_out_width):
        self._obs_out_height = obs_out_height
        self._obs_out_width = obs_out_width

    def get_observation_for_timestep(self, holodeck_state: dict) -> np.ndarray:
        bgr_hwc_color_image: np.ndarray = holodeck_state["DuckiebotsSemanticMaskCamera"][:, :, :3]
        rgb_hwc_color_image = bgr_hwc_color_image[:, :, ::-1]

        rgb_hwc_color_image = cv2.resize(rgb_hwc_color_image,
                                         (self._obs_out_width, self._obs_out_height),
                                         interpolation=cv2.INTER_AREA).astype(dtype=np.uint8)

        return rgb_hwc_color_image


class UEDuckiebotsRotAndLinearVelObservationFunction(UEObservationFunction):

    MAX_ABS_FORWARD_VEL = 60.0  # changed Jan 4. Was 6 but between Oct 22 and Dec, units must have been rescaled in the game.
    #MAX_ABS_FORWARD_VEL = 6.0
    MAX_ABS_YAW_VEL = 159.0

    def get_observation_for_timestep(self, holodeck_state: dict) -> np.ndarray:
        linear_forward_vel: np.ndarray = holodeck_state['DuckiebotsVelocitySensor'][0]
        yaw_vel: np.ndarray = holodeck_state['DuckiebotsVelocitySensor'][1]

        normalized_forward_vel = np.clip(linear_forward_vel,
                                        a_min=-self.MAX_ABS_FORWARD_VEL,
                                        a_max=self.MAX_ABS_FORWARD_VEL) / self.MAX_ABS_FORWARD_VEL

        normalized_rot_yaw_vel = np.clip(yaw_vel,
                                        a_min=-self.MAX_ABS_YAW_VEL,
                                        a_max=self.MAX_ABS_YAW_VEL) / self.MAX_ABS_YAW_VEL

        yaw_and_forward_vel = np.asarray([normalized_rot_yaw_vel, normalized_forward_vel], dtype=np.float32)
        return yaw_and_forward_vel


class UEDuckiebotsPositionAndYawObservationFunction(UEObservationFunction):

    MIN_XY_LOCATION = np.asarray([-115.6, -85.1])
    MAX_XY_LOCATION = np.asarray([107.6, 80.5])

    YAW_MAX_ABSOLUTE_VALUE = 180.0

    @classmethod
    def normalize_xy_yaw(cls, xy_yaw: np.ndarray):
        xy_position = xy_yaw[:2]
        yaw = xy_yaw[2]

        # normalized_xy_location = (np.clip(xy_position,
        #                                   a_min=cls.MIN_XY_LOCATION,
        #                                   a_max=cls.MAX_XY_LOCATION) * 2) / (cls.MAX_XY_LOCATION - cls.MIN_XY_LOCATION)

        normalized_xy_location = (xy_position * 2) / (cls.MAX_XY_LOCATION - cls.MIN_XY_LOCATION)

        normalized_yaw = yaw / cls.YAW_MAX_ABSOLUTE_VALUE

        position_and_yaw = np.asarray([*normalized_xy_location, normalized_yaw], dtype=np.float32)
        return position_and_yaw

    def get_observation_for_timestep(self, holodeck_state: dict) -> np.ndarray:
        xy_position: np.ndarray = holodeck_state['LocationSensor'][:2]
        yaw: np.ndarray = holodeck_state['YawSensor'][0]

        normalized_xy_location = (np.clip(xy_position,
                                        a_min=self.MIN_XY_LOCATION,
                                        a_max=self.MAX_XY_LOCATION) * 2) / (self.MAX_XY_LOCATION - self.MIN_XY_LOCATION)

        normalized_yaw = yaw / self.YAW_MAX_ABSOLUTE_VALUE

        position_and_yaw = np.asarray([*normalized_xy_location, normalized_yaw], dtype=np.float32)
        return position_and_yaw