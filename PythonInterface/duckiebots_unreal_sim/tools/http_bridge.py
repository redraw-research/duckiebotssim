from typing import Optional, Tuple, List

import numpy as np
import requests

DEFAULT_REMOTE_CONTROL_UE_OBJECT_PATH = "/Game/ThirdPerson/Maps/ThirdPersonMap.ThirdPersonMap:PersistentLevel.MyRemoteControlPresetActor_C_UAID_0C9D92856429AD2201_2068398511"


def _parse_rgb_image_from_linear_samples_dict_list(linear_samples: List[dict],
                                                   expected_image_shape_height_width: Tuple[int, int]) -> np.ndarray:
    rgb_samples = []
    for sample in linear_samples:
        rgb_samples.append((sample["R"], sample["G"], sample["B"]))
    image_rgb_out = np.asarray(rgb_samples, dtype=np.int8).reshape((*expected_image_shape_height_width, 3))[::-1, :, :]
    return image_rgb_out


class UEHTTPBridge:

    def __init__(self, server_host: str = "localhost", server_port: int = 30010, protocol: str = "http"):
        self._server_host = server_host
        self._server_port = server_port
        self._protocol = protocol

        self._session = requests.Session()

    @property
    def _base_url(self) -> str:
        return f"{self._protocol}://{self._server_host}:{self._server_port}"

    def ping(self) -> Optional[dict]:
        try:
            r = self._session.get(
                url=f"{self._base_url}/remote/info",
            )
        except requests.exceptions.RequestException as e:
            print(e)
            return None

        if r.ok:
            return r.json()
        return None

    def close(self):
        self._session.close()

    def __del__(self):
        self.close()

    def make_ue_function_call(self,
                              function_name: str,
                              function_parameters: Optional[dict] = None,
                              raise_error: bool = False,
                              object_path: Optional[str] = None) -> Optional[dict]:

        function_parameters = function_parameters or {}

        json_input = {
            "objectPath": object_path or DEFAULT_REMOTE_CONTROL_UE_OBJECT_PATH,
            "functionName": function_name,
            "parameters": function_parameters,
        }

        try:
            r = self._session.put(
                url=f"{self._base_url}/remote/object/call",
                json=json_input
            )
        except requests.exceptions.RequestException as e:
            if raise_error:
                raise
            print(e)
            return None

        if r.ok:
            return r.json()
        return None

    def capture_scene_at_location(self,
                                  location_xyz: Tuple[float, float, float],
                                  roll_pitch_yaw: Tuple[float, float, float]) -> Tuple[
        Optional[np.ndarray], Optional[np.ndarray]]:
        function_parameters = {
            "CameraLocationXYZ": {
                "X": location_xyz[0],
                "Y": location_xyz[1],
                "Z": location_xyz[2]
            },
            "CameraRollDegrees": roll_pitch_yaw[0],
            "CameraPitchDegrees": roll_pitch_yaw[1],
            "CameraYawDegrees": roll_pitch_yaw[2],
        }
        json_result = self.make_ue_function_call(function_name="CaptureSceneAtLocation",
                                                 function_parameters=function_parameters)

        if not json_result:
            return None, None

        rbg_image = _parse_rgb_image_from_linear_samples_dict_list(linear_samples=json_result["RGBOutLinearSamples"],
                                                                   expected_image_shape_height_width=(84, 84))

        mask_image = _parse_rgb_image_from_linear_samples_dict_list(linear_samples=json_result["MaskOutLinearSamples"],
                                                                    expected_image_shape_height_width=(84, 84))

        return rbg_image, mask_image
