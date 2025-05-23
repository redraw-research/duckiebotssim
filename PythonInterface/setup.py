from setuptools import setup

setup(
   name='duckiebots_unreal_sim',
   install_requires=[
      "pyglet==1.5.23",
      "requests",
      "numpy",
      "pyyaml",
      "keyboard",
      "tqdm",
      "termcolor",
      "gym",
      "opencv-python",
      "psutil",
      "inputs",
      "onnxruntime"
   ],
)
