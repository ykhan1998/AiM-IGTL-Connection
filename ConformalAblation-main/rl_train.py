import os

import torch
import yaml
from stable_baselines3 import DQN, PPO
from stable_baselines3.common.env_checker import check_env
from stable_baselines3.common.evaluation import evaluate_policy
from stable_baselines3.common.monitor import Monitor

from matlab_env import MatlabEnv
from rl_callbacks import *


def train(log_dir, num_steps):
    acpr_filename = 'matlab_functions/pressure_fields/P90_100_mm_side_length_alpha_31_5.2373MHz_7.8V_V1.csv.mat'

    env = MatlabEnv(hyper_params, acpr_filename, verbose=False, plot=False)
    check_env(env)
    env = Monitor(env, log_dir)

    model = DQN('CnnPolicy', env)
    # model = PPO('MlpPolicy', env, verbose=0)
    # model.learn(total_timesteps=10000)
    # model.save("conformal_ablation_agent")

    # Create callbacks
    auto_save_callback = SaveOnBestTrainingRewardCallback(check_freq=1000, log_dir=log_dir)
    # plot_callback = PlottingCallback(log_dir=log_dir)

    with ProgressBarManager(num_steps) as progress_callback:
        # This is equivalent to callback=CallbackList([progress_callback, auto_save_callback])
        # model.learn(num_steps, callback=[progress_callback, plot_callback, auto_save_callback])
        model.learn(num_steps, callback=[progress_callback, auto_save_callback])

def plot_results(log_dir):
    from stable_baselines3.common import results_plotter
    results_plotter.plot_results([log_dir], 1e5, results_plotter.X_TIMESTEPS, "Conformal Ablation")

    def moving_average(values, window):
        """
        Smooth values by doing a moving average
        :param values: (numpy array)
        :param window: (int)
        :return: (numpy array)
        """
        weights = np.repeat(1.0, window) / window
        return np.convolve(values, weights, 'valid')


    def plot_results(log_folder, title='Learning Curve'):
        """
        plot the results

        :param log_folder: (str) the save location of the results to plot
        :param title: (str) the title of the task to plot
        """
        x, y = ts2xy(load_results(log_folder), 'timesteps')
        y = moving_average(y, window=50)
        # Truncate x
        x = x[len(x) - len(y):]

        fig = plt.figure(title)
        plt.plot(x, y)
        plt.xlabel('Number of Timesteps')
        plt.ylabel('Rewards')
        plt.title(title + " Smoothed")
        # plt.show()
        plt.savefig(log_dir + 'foo.png')

    plot_results(log_dir)

if __name__ == "__main__":
    use_cuda = torch.cuda.is_available()
    print(f"Using CUDA: {use_cuda}")

    hyper_params_filename = "./hyper_params.yml"
    assert os.path.isfile(hyper_params_filename)
    hyper_params = yaml.load(open(hyper_params_filename), Loader=yaml.FullLoader)
    log_dir = "./tmp/conformalablation/"
    
    num_steps = hyper_params["TRAIN"]['NUM_STEPS']
    print("Start training for " + f"{num_steps}" + " steps.")
    train(log_dir, num_steps)

    plot_results(log_dir)
    pass
