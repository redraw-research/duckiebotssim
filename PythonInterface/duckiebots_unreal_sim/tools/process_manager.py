import os
import subprocess
from typing import Optional
import psutil

from duckiebots_unreal_sim.tools.util import is_windows_os

if is_windows_os():
    DEFAULT_GAME_LAUNCHER_PATH = os.path.join(os.path.dirname(os.path.dirname(__file__)),
                                          "packaged_games", "Windows", "DuckiebotsSim.exe")
else:
    DEFAULT_GAME_LAUNCHER_PATH = os.path.join(os.path.dirname(os.path.dirname(__file__)),
                                          "packaged_games", "Linux", "DuckiebotsSim.sh")

DEFAULT_WEB_APP_LAUNCH_SCRIPT_PATH = os.path.join(os.path.dirname(os.path.dirname(__file__)),
                                          "remote_control_web_app", "WebApp", "Start.sh")


def _kill_recursive(proc_pid: int, only_terminate: bool = True, timeout_seconds: int = 3):
    process = psutil.Process(proc_pid)
    children = process.children(recursive=True)
    if only_terminate:
        process.terminate()
        process.wait(timeout=timeout_seconds)
    for proc in children:
        if only_terminate:
            proc.terminate()
            proc.wait(timeout=timeout_seconds)
        else:
            proc.kill()
    if not only_terminate:
        process.kill()


class UEProcessManager:

    def __init__(self,
                 game_launcher_file_path: str = DEFAULT_GAME_LAUNCHER_PATH,
                 web_app_file_path: str = DEFAULT_WEB_APP_LAUNCH_SCRIPT_PATH):

        if not os.path.exists(game_launcher_file_path):
            raise FileNotFoundError(f"The game launcher file path {game_launcher_file_path} couldn't be found.")

        self._game_launcher_path = game_launcher_file_path
        self._game_process: Optional[subprocess.Popen] = None

        self._web_app_path = web_app_file_path
        self._web_app_process: Optional[subprocess.Popen] = None

    def launch_game(self, render_off_screen: bool = False):
        # launch process with these args to start webserver
        # -RCWebControlEnable -RCWebInterfaceEnable
        launch_args = ["-RCWebControlEnable", "-RCWebInterfaceEnable"]
        if render_off_screen:
            # renders headlessly
            launch_args.append("-RenderOffscreen")

        self._game_process = subprocess.Popen([self._game_launcher_path, *launch_args], shell=False)
        # ^ Don't use shell=True, which treats arguments differently in Windows vs Linux
        # https://stackoverflow.com/questions/20140137/why-does-passing-variables-to-subprocess-popen-not-work-despite-passing-a-list-o?noredirect=1&lq=1

    def launch_node_js_web_control_interface_if_linux(self):
        # Some packaging configurations, especially for Linux, may fail to include the nodejs WebApp or launch it.
        # This launches it manually
        if not is_windows_os():
            self._web_app_process = subprocess.Popen([self._web_app_path], shell=False)

    def join(self):
        if self._game_process and self._game_process.poll():
            self._game_process.wait()

    def close(self):
        if self._game_process:
            _kill_recursive(self._game_process.pid)
            self._game_process.wait()
            self._game_process = None
        if self._web_app_process:
            _kill_recursive(self._web_app_process.pid)
            self._web_app_process.wait()
            self._web_app_process = None

    def __del__(self):
        self.close()
