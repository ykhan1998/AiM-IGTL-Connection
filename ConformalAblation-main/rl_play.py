import os

import torch
import yaml
from stable_baselines3 import DQN
from stable_baselines3.common.env_checker import check_env
from stable_baselines3.common.evaluation import evaluate_policy
from stable_baselines3.common.monitor import Monitor

from matlab_env import MatlabEnv


def play(env, model, log_dir, render=False):
    # check_env(env)
    env = Monitor(env, log_dir)
    obs = env.reset()
    done = False
    while not done:
        action, _states = model.predict(obs, deterministic=True)
        obs, reward, done, info = env.step(action)
        if render:
            env.render(mode='rgb_array')

def eval_agent(model):
    # Evaluate the agent
    # NOTE: If you use wrappers with your environment that modify rewards,
    #       this will be reflected here. To evaluate with original rewards,
    #       wrap environment in a "Monitor" wrapper before other wrappers.
    mean_reward, std_reward = evaluate_policy(model, model.get_env(), n_eval_episodes=10)
    print(f"mean_reward:{mean_reward:.2f} +/- {std_reward:.2f}")


if __name__ == "__main__":
    use_cuda = torch.cuda.is_available()
    print(f"Using CUDA: {use_cuda}")

    hyper_params_filename = "./hyper_params.yml"
    assert os.path.isfile(hyper_params_filename)
    hyper_params = yaml.load(open(hyper_params_filename), Loader=yaml.FullLoader)

    acpr_filename = 'matlab_functions/pressure_fields/P90_100_mm_side_length_alpha_31_5.2373MHz_7.8V_V1.csv.mat'
    env = MatlabEnv(hyper_params, acpr_filename, verbose=True, plot=False)
    model = DQN.load("model/conformal_ablation_agent", env=env)

    print("Model loaded.")

    log_dir = "./tmp/conformalablation/"
    # eval_agent(model)
    play(env, model, log_dir)

    pass
