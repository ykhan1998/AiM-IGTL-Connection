import datetime
import glob
import os
import shutil
import time

import cv2
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import yaml
from gymnasium.wrappers.monitoring.video_recorder import VideoRecorder
from tqdm import tqdm

from agent_dqn import Agent_DQN
from constants import (
    FIG_DIR,
    SAVE_DIR,
    SETTINGS_DIR,
    STATUS_2,
    STATUS_4,
    STATUS_VALUES,
    WORK_DIR,
)
from lesion_pattern import LesionPattern
from matlab_env import MatlabEnv
from utils import compute_radial_diff


def initialize_workspace(env_config_fname, hyper_params_fname):
    assert os.path.isfile(env_config_fname)
    assert os.path.isfile(hyper_params_fname)
    hyper_params = yaml.load(open(hyper_params_fname), Loader=yaml.FullLoader)
    env_config = yaml.load(open(env_config_fname), Loader=yaml.FullLoader)

    datetime_now = datetime.datetime.now().strftime("%Y%m%d-%H-%M-%S")
    
    save_folder = SAVE_DIR + "/" + datetime_now
    
    os.makedirs(save_folder)
    os.makedirs(save_folder + "/traj")

    # Copy env_config and hyper_params
    shutil.copyfile(env_config_fname, save_folder + "/env_config.yml")
    shutil.copyfile(hyper_params_fname, save_folder + "/hyper_params.yml")

    # Create a .csv file for training log
    with open(save_folder + "/log.csv", 'w') as f:
        f.write("Episode,Total Reward,Termination Cause\n")

    return datetime_now, env_config, hyper_params

def train(env_config, hyper_params, datetime_now):
    env = MatlabEnv(env_config, verbose=False, plot=False, render_mode="rgb_array")
    agent = Agent_DQN(env, hyper_params, datetime_now)
    lesion_pattern_obj = LesionPattern(dataset="train")

    for e in range(hyper_params["EPISODES"]):
        print(f"Episode: {e}")
        # Randomly select a lesion map.
        lesion_mask, num_lesion_pixel, seg_fname = lesion_pattern_obj.masking()
        lesion_mask = cv2.resize(lesion_mask, 
                                    dsize=(env.height, env.width),
                                    interpolation=cv2.INTER_NEAREST)
        
        # Randomly select .mat file under the ACPR/rl directory
        # acpr_filename = random.choice(glob.glob(WORK_DIR + "/ACPR/rl/P90*.mat"))
        # Or specify the acpr file
        acpr_filename = WORK_DIR + "/ACPR/rl/P90-20V.mat"

        print(f"Lesion Mask: {seg_fname}, ACPR: {acpr_filename}")

        state = env.reset(lesion_mask, acpr_filename)
        total_reward = 0
        while True:
            action = agent.make_action(state, test=False)
            # action = self.env.action_space.sample() # random action
            next_state, reward, done, info = env.step(action, angle=None, power_switch=True)
            
            total_reward += reward
            agent.push(state, next_state, action, reward, done)
            q, loss = agent.learn()
            state = next_state
            if done:
                termination_cause = info["termination_cause"]
                # create a .csv file
                with open(agent.save_dir + "/log.csv", 'a') as f:
                    f.write(f"{e},{total_reward},{termination_cause}\n")
                break
        if e % agent.test_every == 0:
            agent.save()
            test(env_config, test_save_dir=SAVE_DIR + f"/{datetime_now}",
                    agent=agent, record_video=False)
            with open(agent.save_dir + "/log.txt", 'a') as f:
                f.write(f"Trained Episode: {e}, \
                        Step: {agent.curr_step}, \
                        EXP Rate: {agent.exploration_rate}\n")

def test(env_config, test_save_dir, agent=None, model_path=None, record_video=False):
    print("Starting test procedure...")
    os.makedirs(test_save_dir, exist_ok=True)
    
    env = MatlabEnv(env_config, verbose=False, plot=False, render_mode="rgb_array")

    # lesion_pattern_obj = LesionPattern(dataset="demo")
    lesion_pattern_obj = LesionPattern(dataset="test")
    
    agent = Agent_DQN(env, hyper_params, datetime_now) if agent is None else agent
    if model_path:
        agent.load_trained_model(model_path)
    
    start_time = time.time()
    test_results = {
        'lesion_masks': [],
        'possible_max_rewards': [],
        'rewards': [],
        'termination_causes': [],
        'ablated_tumor_percentages': [],
        'ablated_healthy_percentages': [],
        'radial_diff': []
    }
    num_lesion_masks = len(lesion_pattern_obj.tumor_segs)
    
    print(f"Number of lesion masks to test: {num_lesion_masks}")

    # Enumerate all lesion masks in the test set 
    for seg_idx in tqdm(range(num_lesion_masks)):
        lesion_mask, num_lesion_pixel, seg_fname = lesion_pattern_obj.masking(seg_idx)
        lesion_id = os.path.basename(seg_fname).split(".")[0]
        test_save_dir_lesion = test_save_dir + f"/{lesion_id}"
        os.mkdir(test_save_dir_lesion)

        if record_video:
            vid = VideoRecorder(env, path=test_save_dir_lesion+"/test_vid.mp4", enabled=True)

        lesion_mask = cv2.resize(lesion_mask, 
                                    dsize=(env.height, env.width),
                                    interpolation=cv2.INTER_NEAREST)
        
        # for acpr_filename in glob.glob(WORK_DIR + "/ACPR/rl/P90*.mat"):
        for acpr_filename in glob.glob(WORK_DIR + "/ACPR/rl/P90-20V.mat"):
            state = env.reset(lesion_mask, acpr_filename)
            episode_reward = 0.0
            agent.cur_angle = 0
            done = False
            while not done:
                action = agent.make_action(state, test=True)
                agent.cur_angle += int(action - 1) * env.angle_step
                with open(os.path.join(test_save_dir_lesion, "play_traj.txt"), 'a') as f:
                    f.write(str(agent.cur_angle)+ "\n")

                next_state, reward, done, info = env.step(action, angle=None, power_switch=True)
                episode_reward += reward
                state = next_state

                if record_video:
                    vid.capture_frame()

                if done:
                    # Process and save the final state
                    status_map = next_state[2]
                    ablated_tumor_idx = status_map == STATUS_4
                    ablated_healthy_idx = status_map == STATUS_2
                    
                    num_lesion_pixels = np.count_nonzero(lesion_mask)
                    ablated_tumor_percent = np.count_nonzero(ablated_tumor_idx) / num_lesion_pixels * 100
                    ablated_healthy_percent = np.count_nonzero(ablated_healthy_idx) / num_lesion_pixels * 100

                    radial_diff = compute_radial_diff(status_map)
                    
                    # Create ablation pattern
                    ablation_pattern = np.zeros_like(status_map)
                    ablation_pattern[ablated_tumor_idx | ablated_healthy_idx] = 1

                    # Save numpy arrays
                    np.save(os.path.join(test_save_dir_lesion, "status_map.npy"), status_map)
                    np.save(os.path.join(test_save_dir_lesion, "lesion_mask.npy"), lesion_mask)
                    np.save(os.path.join(test_save_dir_lesion, "ablation_pattern.npy"), ablation_pattern)
                    
                    plt.imshow(status_map)
                    plt.axis('off')                    
                    plt.savefig(os.path.join(test_save_dir_lesion, "status_map.png"), bbox_inches='tight', pad_inches=0)
                    plt.close()

                    # Append results to lists
                    test_results['lesion_masks'].append(lesion_id)
                    test_results['possible_max_rewards'].append(num_lesion_pixel * STATUS_VALUES["lesion_ablated"])
                    test_results['rewards'].append(episode_reward)
                    test_results['termination_causes'].append(info["termination_cause"])
                    test_results['ablated_tumor_percentages'].append(ablated_tumor_percent)
                    test_results['ablated_healthy_percentages'].append(ablated_healthy_percent)
                    test_results['radial_diff'].append(radial_diff)

                    # Close video recording if enabled
                    if record_video:
                        vid.close()
                        vid.enabled = False
                        
                    break

    test_res_df = pd.DataFrame(test_results)

    csv_path = os.path.join(test_save_dir, "test_results.csv")
    test_res_df.to_csv(csv_path, index=False)

    mean_reward = np.mean(test_results['rewards'])

    print(f"Tested {num_lesion_masks} episodes. Mean reward: {mean_reward}. Time: {time.time()-start_time}.")

    mean_radial_diff = np.mean(test_results['radial_diff'])
    print(f"Mean radial difference: {mean_radial_diff}")
    
    env.close()

    return np.mean(test_results['rewards'])


if __name__ == '__main__':
    env_config_fname = SETTINGS_DIR + "/env_config.yml"
    hyper_params_fname = SETTINGS_DIR + "/hyper_params.yml"

    datetime_now, env_config, hyper_params = initialize_workspace(env_config_fname, hyper_params_fname)

    train(env_config, hyper_params, \
            datetime_now=datetime_now)

    # i = 0
    # for dqn_fname in sorted(glob.glob(SAVE_DIR + "/20240223-14-39-31/net_*.chkpt"), key=os.path.getmtime):
    #     if i % 5 == 0:
    #         print(f"Testing {dqn_fname}")
    #         test(env_config, traj_save_dir=SAVE_DIR + f"/{datetime_now}/traj/{i*500}", \
    #             model_path=dqn_fname, record_video=True)
    #     i += 1
    
    # DQN_path = r'/home/yiwei/ConformalAblation/save/20240223-14-39-31/net_90605.chkpt' # 100
    # DQN_path = r'/home/yiwei/ConformalAblation/save/20240223-14-39-31/net_2040930.chkpt' # 5100
    # DQN_path = r'/home/yiwei/ConformalAblation/save/20240223-14-39-31/net_3369042.chkpt' # 10200
    # DQN_path = r'/home/yiwei/ConformalAblation/save/20240223-14-39-31/net_4205344.chkpt' # 13500
    
    # test(env_config,
    #         test_save_dir=WORK_DIR + f"/test/{datetime_now}", \
    #         model_path=DQN_path, \
    #         record_video=True)
        
