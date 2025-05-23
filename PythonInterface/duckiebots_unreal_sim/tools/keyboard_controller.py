import keyboard

from duckiebots_unreal_sim.tools.util import is_windows_os

# TODO (author1) On linux, the 'keyboard' module requires root, which is a silly requirement.
# TODO Can we make this work with another package?

def get_keyboard_turning() -> float:
    try:
        turn = 0.0
        if keyboard.is_pressed("right arrow"):
            turn += 1.0
        if keyboard.is_pressed("left arrow"):
            turn -= 1.0
    except ImportError:
        if not is_windows_os():
            print("On linux, the 'keyboard' module requires root, which is a silly requirement. "
                  "This needs to be implemented differently.")
        raise
    return turn


def get_keyboard_velocity() -> float:
    try:
        vel = 0.0
        if keyboard.is_pressed("up arrow"):
            vel += 1.0
        if keyboard.is_pressed("down arrow"):
            vel -= 1.0
    except ImportError:
        if not is_windows_os():
            print("On linux, the 'keyboard' module requires root, which is a silly requirement. "
                  "This needs to be implemented differently.")
        raise
    return vel
