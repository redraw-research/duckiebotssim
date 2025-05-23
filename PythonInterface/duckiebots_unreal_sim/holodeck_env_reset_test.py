import time

from duckiebots_unreal_sim.holodeck_env import UEDuckiebotsHolodeckEnv, DEFAULT_GAME_LAUNCHER_PATH
import numpy as np

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
                                  physics_hz=10 if locked_physics_rate else None)

    env2 = UEDuckiebotsHolodeckEnv(launch_game_process=launch_game_process,
                                  render_game_on_screen=render_launched_game_on_screen,
                                  randomization_enabled=False,
                                  physics_hz=10 if locked_physics_rate else None)

    np.set_printoptions(edgeitems=10)
    np.core.arrayprint._line_width = 400

    max_diff = None
    all_diffs = []
    print("env initialized")
    while True:
        i = 0
        env.reset()
        env2.reset()
        # env.step(action=np.asarray([0.0, 0.0], dtype=np.float32), override_duckiebot_state=[14.945293, 10.28613, -49.6523, 0., 0.])

        env2.step(action=np.asarray([0.0, 0.0], dtype=np.float32), override_duckiebot_state=[14.945293, 10.28613, -49.6523, 0., 0.])
        env2.step(action=np.asarray([0.0, 0.0], dtype=np.float32), override_duckiebot_state=env.get_duckiebot_state())

        # env.render()
        # env2.render()

        # These lines return the color image and mask (as BGR pixels) as np.ndarrays:
        # color_image = env.render(mode="rgb_array")
        # mask_image = env.render_mask(mode="rgb_array")

        episode_reward = 0.0
        done = False
        while not done:
            # input("press enter to progress")
            # print("Begin frame ---------------------------------------------------------")

            velocity = gamepad.LeftJoystickY if gamepad else get_keyboard_velocity()
            turning = gamepad.LeftJoystickX if gamepad else get_keyboard_turning()
            action = np.asarray([velocity, turning], dtype=np.float32)

            # Example to randomly select an action in the environment:
            # action = env.action_space.sample()
            # print(f"duckiebot state: {env.get_duckiebot_state()}")

            # obs2, rew2, done2, info2 = env2.step(action=np.zeros_like(action), override_duckiebot_state=env.get_duckiebot_state())
            # obs2, rew2, done2, info2 = env2.step(action=action)

            obs, rew, done, info = env.step(action=action)
            obs2, rew2, done2, info2 = env2.step(action=action, override_duckiebot_state=env.get_duckiebot_state())

            # env2.step(action=action)
            state1 = done
            state2 = done2
            # state1 = env.get_duckiebot_state()
            # state2 = env2.get_duckiebot_state()
            # state1 = np.asarray([rew])
            # state2 = np.asarray([rew2])
            
#             print(f"duckiebot state: {state1}\n"
#                   f"state 2:         {state2}\n"
#                   f"diff             {state1 - state2}")
#             diff = np.abs(state1 - state2)
            diff = state1 != state2

            all_diffs.append(diff)
            if max_diff is None:
                max_diff = np.zeros_like(diff)
            max_diff = np.max([max_diff, diff], axis=0)
            print(f"1: {state1:.3f}, 2: {state2:.3f} diff {diff} max: {max_diff}")

            # print(f"1: {state1:.3f}, 2: {state2:.3f} diff {diff} max: {max_diff} mean: {np.percentile(all_diffs, axis=0, q=95)}")
            # print("End frame ---------------------------------------------------------")
            # env2.step(action=np.asarray([0.0, 0.0], dtype=np.float32),
            #           override_duckiebot_state=[14.945293, 10.28613, -49.6523, 0., 0.])

            episode_reward += rew

            env.render()
            env2.render()

            # These lines return the color image and mask (as BGR pixels) as np.ndarrays:
            # color_image = env.render(mode="rgb_array")
            # mask_image = env.render_mask(mode="rgb_array")

            if gamepad and gamepad.B:
                print("randomizing")
                # env.reset()
                env.randomize()

            i += 1
            # time.sleep((0.05))
        print(f"episode reward: {episode_reward}")
