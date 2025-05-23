import os

import numpy as np

from duckiebots_unreal_sim.tools import ImageRenderer

if __name__ == '__main__':

    # put your own rgb frame directory here
    example_frame_directory = r"C:\Users\author1\Documents\GitHub\duckiebotssim\PythonInterface\duckiebots_unreal_sim\data\frame"

    example_segment_directory = os.path.join(os.path.dirname(example_frame_directory), "segment")

    image_renderer = ImageRenderer(height=84, width=84)
    image_renderer2 = ImageRenderer(height=84, width=84)

    for filename in os.listdir(example_frame_directory):
        frame_abs_path = os.path.join(example_frame_directory, filename)
        segment_abs_path = os.path.join(example_segment_directory, filename)

        if filename.endswith(".npy") and os.path.isfile(segment_abs_path):
            frame_rgb = np.load(frame_abs_path).astype(np.uint8)  # images should be loaded as uint8 from 0 to 255
            assert np.min(frame_rgb) >= 0
            assert np.max(frame_rgb) <= 255
            # image channels ordered are red, green, blue (RGB), but cv2 expects BGR
            frame_bgr = frame_rgb[:, :, ::-1]
            image_renderer.render_cv2_image(cv2_image=frame_bgr)

            segment_rgb = np.load(segment_abs_path).astype(np.uint8)  # images should be loaded as uint8 from 0 to 255
            assert np.min(segment_rgb) >= 0
            assert np.max(segment_rgb) <= 255
            # image channels ordered are red, green, blue (RGB), but cv2 expects BGR
            segment_bgr = segment_rgb[:, :, ::-1]
            image_renderer2.render_cv2_image(cv2_image=segment_bgr)

        input("Press Enter to go to the next pair of images.")

    print("Done.")
