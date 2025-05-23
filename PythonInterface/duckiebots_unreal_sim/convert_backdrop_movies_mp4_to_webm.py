import argparse
import glob
import os
import subprocess

from termcolor import colored
from tqdm import tqdm

# (author1) Not currently used. Instead, we use the Bink bk2 video format.
# https://docs.unrealengine.com/5.0/en-US/bink-video-for-unreal-engine/

_DEFAULT_BACKDROP_MOVIE_PATH = os.path.join(os.path.dirname(__file__), "backdrop_movies")

if __name__ == '__main__':
    # creates a copy of every mp4 in the backdrop_movie_directory
    parser = argparse.ArgumentParser()
    parser.add_argument("-movie_dir", type=str, default=_DEFAULT_BACKDROP_MOVIE_PATH, required=False,
                        help="Directory containing mp4 files to make webm copies of.")
    parser.add_argument("-cpu_only", action="store_true", required=False,
                        help="Don't tell ffmpeg to use CUDA GPU acceleration when encoding video.")
    args = parser.parse_args()
    backdrop_movie_directory = args.movie_dir
    cpu_only = args.cpu_only

    print(colored(f"Backdrop movie directory is {backdrop_movie_directory}.", "blue"))

    if not os.path.isdir(backdrop_movie_directory):
        raise NotADirectoryError(f"{backdrop_movie_directory} isn't a directory.")

    mp4_file_paths = glob.glob(f"{backdrop_movie_directory}/*.mp4")
    print(colored(f"{len(mp4_file_paths)} mp4 files found.", "blue"))

    if cpu_only:
        use_gpu_arg = ""
        print(colored(f"Using CPU only to encode videos.", "green"))
    else:
        # https://stackoverflow.com/questions/44510765/gpu-accelerated-video-processing-with-ffmpeg
        use_gpu_arg = "-hwaccel cuda "  # (author1) I'm not sure how much help this is actually doing.
        print(colored(f"Trying to use Nvidia GPU to encode videos.", "green"))

    with tqdm(list(enumerate(mp4_file_paths)), desc=colored("Movies Converted:", "blue"), colour="green") as movies_bar:
        for i, mp4_path in movies_bar:
            # https://video.stackexchange.com/questions/19590/convert-mp4-to-webm-without-quality-loss-with-ffmpeg
            print(colored(f"Pass 1 of 2 on {mp4_path} (movie {i + 1}/{len(mp4_file_paths)}):", "blue"))
            command_1 = f"ffmpeg {use_gpu_arg}-i {mp4_path} -b:v 0 -crf 30 -pass 1 -an -f webm -y /dev/null"
            print(colored(f"running {command_1.split(' ')}", "yellow"))
            subprocess.run(command_1.split(" "), shell=False, check=True)

            print(colored(f"Pass 2 of 2 on {mp4_path} (movie {i + 1}/{len(mp4_file_paths)}):", "blue"))
            command_2 = f"ffmpeg {use_gpu_arg}-i {mp4_path} -b:v 0 -crf 30 -pass 2 -y {mp4_path[:-4]}.webm"
            print(colored(f"running {command_2.split(' ')}", "yellow"))
            subprocess.run(command_2.split(" "), shell=False, check=True)

    print(colored("Done!", "blue"))
