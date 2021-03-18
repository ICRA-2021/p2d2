# R3L: Rapidly Randomly-exploring Reinforcement Learning

This folder contains code to run R3L, a specific algorithm in the P2D2 paradigm.

## Dependencies
You must first install Mujoco; see instructions [here](https://github.com/openai/mujoco-py#install-mujoco).
(On Ubuntu, you may need to run the following command `sudo apt install libosmesa6-dev libgl1-mesa-glx libglfw3`).

Then, there are two alternatives to install dependencies:
1. Use the provided `environment.yml` file to create a conda environment:
```
conda env create -f environment.yml
```
2. OR Install dependencies manually with `pip`:
```
torch
numpy
matplotlib
gym
mujoco-py
tqdm
ghalton
```

## Running examples
1. `main.py` allows to play with R3L exploration on different RL environments, changing parameters:
    * To try different environments, `env_name` can be changed to one of the following values: `PendulumSparse-v0`, `MountainCarSparse-v0`, `AcrobotContinuous-v0`, `CartpoleSwingup-v0`, `ReacherSparse-v0`.
    * The number of R3L exploration steps `n_iter` can be changed (`10000` is a good value for most environments)
    * The probability to sample the goal is `rrt_goal_sample_rate=0.05` by default, and can be set to `0.0` to disable goal biasing.
2. Generating trajectories for a specific environment and storing them in a file can be achieved with the following command:
```
python record_trajectories -env MountainCarSparse-v0 --n_traj=20 --filepath="trajectories.npz"
```
generated trajectories can then be used to initialize an RL method in the framework of your choice.
To replicate our results, see the `garage/README.md` file.


## Generating tables, figures, and videos
Scripts used to generate figures/results in the paper are located in utils.
* `utils/comparison_r3l_exploration.py` is used to generate results in Table 1.
* `utils/exploration_video_gen.py` is used to generate the R3L exploration tree video.
* `utils/generate_cost_surface.py` is used to generate the MountainCar cost surface in Figure 1,
and `results/plotting_chain_and_loss/plot_it.py` is used to generate the actual figure (background and trajectories).
