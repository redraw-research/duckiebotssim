# DuckiebotsSim


![simulation image](duckiebots_sim_overview.png)

Unreal Engine 5 Duckiebots Simulation

Revised code coming soon!

# Installation

While the Duckiebots simulation can be packaged for and deployed to headless Linux servers, it's recommended to initially build and package the simulation in Windows where the Unreal Engine Editor is better supported.

1. Change working directory to this repo
```bash
git clone https://github.com/redraw-research/duckiebotssim.git
cd duckiebotssim
```

2. Clone our custom UE 5 fork of the [holodeck](https://github.com/BYU-PCCL/holodeck) dependency into an adjacent directory to this one.
```bash
git clone https://github.com/redraw-research/holodeck ../holodeck
```

3. pip install it.
```bash
pip install -e ../holodeck
```

4. pip install the duckiebots_unreal_sim package in this repo
```bash
cd PythonInterface
pip install -e .
```


5. You can open the [DuckiebotsSim.uproject](DuckiebotsSim.uproject) file with the [Unreal Engine 5.3 Editor](https://www.unrealengine.com/en-US/).
Add the free external dependency, the [Luma AI UE plugin](https://www.fab.com/listings/b52460e0-3ace-465e-a378-495a5531e318) in order to render gaussian splats in Unreal Engine.
In the Unreal Editor, package the game for Linux or Windows. Place the contents of the packaged game (a folder called `Linux` or `Windows` depending on your target platform) in the [PythonInterface/duckiebots_unreal_sim/packaged_games](PythonInterface/duckiebots_unreal_sim/packaged_games) directory. After doing so, python code will expect to be able to find the file: `/packaged_games/Linux/DuckiebotsSim.sh` (or on Windows: `/packaged_games/Windows/DuckiebotsSim.exe`)


6. Try out the [holodeck_lane_following_env.py](PythonInterface%2Fduckiebots_unreal_sim%2Fholodeck_lane_following_env.py) keyboard/xbox controller demo
```angular2html
cd duckiebots_unreal_sim
python holodeck_lane_following_env.py
```
