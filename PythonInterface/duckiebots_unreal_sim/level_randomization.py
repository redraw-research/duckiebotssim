import glob
import os
import random
import time
from typing import Optional, Tuple

from duckiebots_unreal_sim.tools import UEHTTPBridge


def randomize_level(http_bridge: UEHTTPBridge,
                    randomize_movie: bool = True,
                    backdrop_movie_directory: Optional[str] = None,
                    road_movie_directory: Optional[str] = None) -> Tuple[bool, bool]:
    movie_file_type = "bk2"

    if backdrop_movie_directory is None:
        backdrop_movie_directory = os.path.join(os.path.dirname(os.path.abspath(__file__)), "backdrop_movies")

    backdrop_movie_swap_result = None
    if backdrop_movie_directory:
        if not os.path.isdir(backdrop_movie_directory):
            raise NotADirectoryError(f"{backdrop_movie_directory} isn't a directory.")

        movie_file_paths = glob.glob(f"{backdrop_movie_directory}/*.{movie_file_type}")
        if len(movie_file_paths) > 0:
            random_movie_path = random.choice(movie_file_paths)
            movie_swap_result = http_bridge.make_ue_function_call(function_name="SetBackdropMovieFilePath",
                                                                  function_parameters={
                                                                      "MoviePath": random_movie_path
                                                                  })
        else:
            print(f"No {movie_file_type} files found in {backdrop_movie_directory}.")

    if road_movie_directory is None:
        road_movie_directory = os.path.join(os.path.dirname(__file__), "road_movies")

    road_movie_swap_result = None
    if road_movie_directory:
        if not os.path.isdir(road_movie_directory):
            raise NotADirectoryError(f"{road_movie_directory} isn't a directory.")

        movie_file_paths = glob.glob(f"{road_movie_directory}/*.{movie_file_type}")
        if len(movie_file_paths) > 0:
            random_movie_path = random.choice(movie_file_paths)
            movie_swap_result = http_bridge.make_ue_function_call(function_name="SetRoadMovieFilePath",
                                                                  function_parameters={
                                                                      "MoviePath": random_movie_path
                                                                  }, raise_error=True)
        else:
            print(f"No {movie_file_type} files found in {road_movie_directory}.")

    randomize_result = http_bridge.make_ue_function_call(function_name="RandomizeLevel",
                                                         function_parameters={})
    return (randomize_result is not None and
            ((backdrop_movie_swap_result is not None and
              road_movie_swap_result is not None) or not randomize_movie))


if __name__ == '__main__':
    http_bridge = UEHTTPBridge()

    while True:
        print("randomizing")
        randomize_level(http_bridge=http_bridge)
        time.sleep(5)
