# ROL ( Removable Order Learning ) with Reinforcement learning

This repo contains the code of the paper "A Novel Ordering-Based Approach for Learning Bayesian Networks in the Presence
of Unobserved Variables" based on the garage code [repository](https://github.com/rlworkgroup/garage) verion 2020.9.0

[Garage docs](https://garage.readthedocs.io/en/latest/)

## Installation

- Download the repo
- It is not needed to install Mujoco
- Activate the enviroment:
  `conda activate myenv`
- Follow garage installation guide for developers and finally go to the garage directory and type(note that you should
  install our modified garage library):
  `pip install -e '.[all,dev]'`
- Finally install the rest of requirements: `pip install -r requirements.txt`


* * *

### How to run?


You can run `run.py` with your desired arguments.
For example:

 `python run.py --alg_name vpg --graph_name sachs --IS_MAG True --data_num 550 --log_path /root/Data/log_graphs --dataset_path /root/Data/sl_data`

To run VPG ( Vanilla Policy gradient) run `python demo_vpg.py` (Please note that you should prepare the dataset and set
the path in the folder)

### Main algorithm:

`algos.py`: our implementation of RL algorithms ( our modified VPG and Value Iteration) of the RL agent in
experiments/.

`GraphEnv.py`: our implementation of Environment of the RL agent in garage/src/garage/envs/.

`ArgMaxMLPPolicy.py`: our implementation of the softmax Policy we defined in the folder
garage/src/garage/torch/policies.

`run.py`: The main code of running VPG or VI algorithm on graphs in the folder experiments/.

`graph_utils.py`: Utilities on graphs (loads, save, CI tests, data generating, plotting).

`tests.py`: Unit tests to check the correctness of coding for the GraphEnvs and Utilities.

`SGDH.py`: the implementation of our algorithm that uses our optimizer in garage/torch/algos/.



