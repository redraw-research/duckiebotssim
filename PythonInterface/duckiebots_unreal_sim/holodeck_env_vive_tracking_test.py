import time

from duckiebots_unreal_sim.holodeck_env import UEDuckiebotsHolodeckEnv, DEFAULT_GAME_LAUNCHER_PATH
import numpy as np

if __name__ == '__main__':
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
                                  physics_hz=10 if locked_physics_rate else None)

    np.set_printoptions(edgeitems=10)
    np.core.arrayprint._line_width = 400

    max_diff = None
    all_diffs = []
    print("env initialized")
    while True:
        env.reset()

        episode_reward = 0.0
        done = False
        while not done:
            velocity = 0
            turning = 0
            action = np.asarray([velocity, turning], dtype=np.float32)

            obs, rew, done, info = env.step(action=action)
            obs, rew, done, inf = env.step(action=action, override_duckiebot_state=env.get_duckiebot_state())



            env.render()


