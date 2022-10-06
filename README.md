# opt-mimic-traj-opt

Trajectory Optimization code for the paper "OPT-Mimic: Imitation of Optimized Trajectories for Dynamic Quadruped Behaviors".

## Quick Reference
- `traj_opt.py` contains the core implementation of the trajectory optimization, including some constraints that need to be modified specifically to each motion.
- `generate_reference.py` contains code producing hard-coded kinematic sketch trajectories via sinusoids and interpolation of keypoints. The motion type is specified using the `motion_type` variable (line 41 as of writing).
- `python main.py -d 0 -s 1 -e 1 -n my-experiment-name` is a typical command I would run to run trajectory optimization, and save video and csv results under filename `my-experiment-name`. You likely need to create an empty `videos/` folder to avoid getting an error the first time.
- `constants.py` contains parameter values, of which I occasionally had to tune LQR weights specifically to each motion.

## Integration with Other Code
- Once a csv trajectory file is saved onto `csv/`, it should be copied onto the `raisimGymTorch/raisimGymTorch/env/envs/solo8_env/traj/` directory of the [reinforcement learning code](https://github.com/yunifuchioka/opt-mimic-raisim) to use it to train RL policies.
- When deploying the trained RL policy on the robot, the csv trajectory file also needs to be copied onto the workspace directory containing the [robot deployment code](https://github.com/yunifuchioka/opt-mimic-robot-deploy). See that repo for details.