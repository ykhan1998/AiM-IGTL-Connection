from datetime import datetime

import matplotlib.pyplot as plt
import numpy as np

from constants import FIG_DIR


def plot_testing_rewards_vs_trained_episodes(log_fname):
    max_episode = 14000

    with open(log_fname, 'r') as f:
        rewards = []
        episodes = []
        for line in f:
            if line.startswith("Trained Episode:"):
                episode = int(line.split("Trained Episode:")[1].split(",")[0].strip())
                if episode > max_episode:
                    episodes.append(episode)
                    break
                episodes.append(episode)
            elif line.startswith("Tested"):
                reward = float(line.split("Mean reward:")[1].split(".")[0].strip())
                rewards.append(reward)
          
    plt.plot(episodes, rewards)
    plt.xlabel('Trained Episodes')
    plt.ylabel('Testing Reward')
    plt.title('Testing Reward vs. Trained Episodes')

    plt.savefig(FIG_DIR + f"/testing_rewards_vs_trained_episodes_{datetime.now().strftime('%Y%m%d-%H-%M-%S')}.png")

def plot_traj(filename):
    angle_setpoints = []
    timestamps = []
    with open(filename) as f:
        lines = f.readlines()
        time = 0
        for line in lines:
            line = line[:-1]
            angle = float(line.split(',')[0])
            time = float(line.split(',')[1])

            angle_setpoints.append(angle)
            timestamps.append(time)
    
    plt.plot(angle_setpoints, timestamps)
    plt.xlabel("Time (s)")
    plt.ylabel("Angle (degree)")
    plt.title("Angle Setpoint vs. Time")
    plt.savefig(FIG_DIR + f"/{datetime.now().strftime('%Y%m%d-%H-%M-%S')}.png")

def plot_sensitivity():

    # create two figures
    plt.figure(1)
    t_ls_sim_ref = np.load("/home/yiwei/ConformalAblation/iCAP/traj_save/20231116-23-35-15/time_curve.npy")
    sab_ls_sim_ref = np.load("/home/yiwei/ConformalAblation/iCAP/traj_save/20231116-23-35-15/cem_curve.npy")
    print(sab_ls_sim_ref[140])
    plt.scatter(t_ls_sim_ref, sab_ls_sim_ref, color='r', s=0.5)
    plt.plot(t_ls_sim_ref, sab_ls_sim_ref, color='r', label="Default")

    t_ls_sim_alpha50 = np.load("/home/yiwei/ConformalAblation/iCAP/traj_save/20240130-00-32-44/time_curve.npy")
    sab_ls_sim_alpha50 = np.load("/home/yiwei/ConformalAblation/iCAP/traj_save/20240130-00-32-44/cem_curve.npy")
    print(sab_ls_sim_alpha50[140])
    plt.scatter(t_ls_sim_alpha50, sab_ls_sim_alpha50, color='orange', s=0.5)
    plt.plot(t_ls_sim_alpha50, sab_ls_sim_alpha50, color='orange', label="Absorption: -15%")

    t_ls_sim_alpha55 = np.load("/home/yiwei/ConformalAblation/iCAP/traj_save/20240130-00-35-43/time_curve.npy")
    sab_ls_sim_alpha55 = np.load("/home/yiwei/ConformalAblation/iCAP/traj_save/20240130-00-35-43/cem_curve.npy")
    print(sab_ls_sim_alpha55[140])
    plt.scatter(t_ls_sim_alpha55, sab_ls_sim_alpha55, color='gold', s=0.5)
    plt.plot(t_ls_sim_alpha55, sab_ls_sim_alpha55, color='gold', label="Absorption: +15%")
    
    t_ls_sim_sos1473 = np.load("/home/yiwei/ConformalAblation/iCAP/traj_save/20231214-13-48-26/time_curve.npy")
    sab_ls_sim_sos1473 = np.load("/home/yiwei/ConformalAblation/iCAP/traj_save/20231214-13-48-26/cem_curve.npy")
    print(sab_ls_sim_sos1473[140])
    plt.scatter(t_ls_sim_sos1473, sab_ls_sim_sos1473, color='springgreen', s=0.5)
    plt.plot(t_ls_sim_sos1473, sab_ls_sim_sos1473, color='springgreen', label="Speed of Sound: -5%")

    t_ls_sim_sos1628 = np.load("/home/yiwei/ConformalAblation/iCAP/traj_save/20231214-13-51-14/time_curve.npy")
    sab_ls_sim_sos1628 = np.load("/home/yiwei/ConformalAblation/iCAP/traj_save/20231214-13-51-14/cem_curve.npy")
    print(sab_ls_sim_sos1628[140])
    plt.scatter(t_ls_sim_sos1628, sab_ls_sim_sos1628, color='green', s=0.5)
    plt.plot(t_ls_sim_sos1628, sab_ls_sim_sos1628, color='green', label="Speed of Sound: +5%")

    plt.xlim(0, 160)
    plt.xlabel("Time (s)")
    plt.ylabel("Ablated Area (mm^2)")
    plt.legend()
    plt.savefig('/home/yiwei/ConformalAblation/fig' + "/s_ab_curve1.png")

    # plt.figure(2)

    # t_ls_sim_ref = np.load("/home/yiwei/ConformalAblation/iCAP/traj_save/20231116-23-35-15/time_curve.npy")
    # sab_ls_sim_ref = np.load("/home/yiwei/ConformalAblation/iCAP/traj_save/20231116-23-35-15/cem_curve.npy")
    # plt.scatter(t_ls_sim_ref, sab_ls_sim_ref, color='r', s=0.5)
    # plt.plot(t_ls_sim_ref, sab_ls_sim_ref, color='r', label="Default")

    # t_ls_sim_tc050 = np.load("/home/yiwei/ConformalAblation/iCAP/traj_save/20231214-13-54-19/time_curve.npy")
    # sab_ls_sim_tc050 = np.load("/home/yiwei/ConformalAblation/iCAP/traj_save/20231214-13-54-19/cem_curve.npy")
    # print(sab_ls_sim_tc050[140])
    # plt.scatter(t_ls_sim_tc050, sab_ls_sim_tc050, color='violet', s=0.5)
    # plt.plot(t_ls_sim_tc050, sab_ls_sim_tc050, color='violet', label="Thermal Conductivity: -5%")

    # t_ls_sim_tc056 = np.load("/home/yiwei/ConformalAblation/iCAP/traj_save/20231214-13-56-35/time_curve.npy")
    # sab_ls_sim_tc056 = np.load("/home/yiwei/ConformalAblation/iCAP/traj_save/20231214-13-56-35/cem_curve.npy")
    # print(sab_ls_sim_tc056[140])
    # plt.scatter(t_ls_sim_tc056, sab_ls_sim_tc056, color='indigo', s=0.5)
    # plt.plot(t_ls_sim_tc056, sab_ls_sim_tc056, color='indigo', label="Thermal Conductivity: +5%")

    # t_ls_sim_hc3278 = np.load("/home/yiwei/ConformalAblation/iCAP/traj_save/20231214-13-59-15/time_curve.npy")
    # sab_ls_sim_hc3278 = np.load("/home/yiwei/ConformalAblation/iCAP/traj_save/20231214-13-59-15/cem_curve.npy")
    # print(sab_ls_sim_hc3278[140])
    # plt.scatter(t_ls_sim_hc3278, sab_ls_sim_hc3278, color='aqua', s=0.5)
    # plt.plot(t_ls_sim_hc3278, sab_ls_sim_hc3278, color='aqua', label="Heat Capacity: -5%")

    # t_ls_sim_hc3623 = np.load("/home/yiwei/ConformalAblation/iCAP/traj_save/20231214-14-01-34/time_curve.npy")
    # sab_ls_sim_hc3623 = np.load("/home/yiwei/ConformalAblation/iCAP/traj_save/20231214-14-01-34/cem_curve.npy")
    # print(sab_ls_sim_hc3623[140])
    # plt.scatter(t_ls_sim_hc3623, sab_ls_sim_hc3623, color='blue', s=0.5)
    # plt.plot(t_ls_sim_hc3623, sab_ls_sim_hc3623, color='blue', label="Heat Capacity: +5%")

    # plt.xlim(0, 160)
    # plt.xlabel("Time (s)")
    # plt.ylabel("Ablated Area (mm^2)")
    # plt.legend()
    # plt.savefig('/home/yiwei/ConformalAblation/fig' + "/s_ab_curve2.png")

    
if __name__ == '__main__':
    # log_fname = r'/home/yiwei/ConformalAblation/save/20230828-16-22-15/log.txt'
    # train_rewards(log_fname)

    # plot_traj(r'/home/yiwei/ConformalAblation/iCAP/traj_save/20231026-22-50-02/traj.txt')

    # plot_sensitivity()

    plot_testing_rewards_vs_trained_episodes(r'/home/yiwei/ConformalAblation/save/20240223-14-39-31/log.txt')