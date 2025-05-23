rsync -vr \
--exclude rllib/data \
--exclude exp \
--exclude exp_plot \
--exclude dbc/.git \
--exclude event_files \
--exclude .idea \
--exclude .vscode \
--exclude .ipynb_checkpoints \
--exclude __pycache__ \
--exclude lightning_logs \
--exclude gpu.lock \
--exclude '*.cache' \
--exclude PythonInterface/duckiebots_unreal_sim/packaged_games/Linux/DuckiebotsSim/Saved/Logs \
~/git/duckiebotssim/PythonInterface arcus-13:~/git/duckiebotssim

rsync -vr \
~/git/holodeck arcus-13:~/git/