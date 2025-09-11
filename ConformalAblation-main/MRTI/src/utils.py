import copy
from os.path import exists
from random import random

import matplotlib.pyplot as plt
import numpy as np


def update_cem(base_temp, cem_map, thermal_img, time_step):
        # t = time.time()
        for i in range(thermal_img.shape[0]):
            for j in range(thermal_img.shape[1]):
                T = thermal_img[i][j]
                if T >= base_temp:
                    R = 0.5
                else:
                    R = 0.25
                cem_map[i][j] += R ** (base_temp - T) * time_step / 60 # mintues
        # elapsed = time.time() - t
        # print(elapsed)
        return cem_map

def viz_mrti_results(results_folder, num_slice, roi, vmin, vmax, 
                        res_idx, pt, plot):
    if plot:
        fig, ax = plt.subplots(1, num_slice)
        plt.cla()
        plt.subplots_adjust(left=0.01, bottom=0.05, right=0.99, top=0.95, \
                            wspace=0.03, hspace=0.2)

    i = 0
    latest_mrti = [0] * num_slice
    cbar = []
    sample_temp = []
    sample_time = []

    with open(results_folder + "/timestamps.txt") as f:
        timestamps = f.readlines()

    while True:
        latest_filename = results_folder + '/' + str(i + 1) + ".npy"
        if exists(latest_filename):
            latest_npy = np.load(latest_filename)
            if not roi:
                roi = [0, 0, latest_npy.shape[0], latest_npy.shape[1]]
            latest_mrti[i % num_slice] = latest_npy[roi[0]:roi[2], roi[1]:roi[3]]
            
            cur_time = float(timestamps[i].split(',')[1][:-1])
            
            if i % num_slice == res_idx:
                sample_temp.append(latest_mrti[i % num_slice][pt[0], pt[1]])
                sample_time.append(cur_time)

            if plot:
                im = ax[i % num_slice].imshow(latest_mrti[i % num_slice], 
                                                vmin=vmin, vmax=vmax)
                ax[i % num_slice].set_title("Slicer:" + str(i % num_slice + 1) + \
                                            " Time: " + str(cur_time))
                ax[i % num_slice].axis('off')
                plt.pause(0.2)
                if i < num_slice:
                    cbar.append(plt.colorbar(im, ax=ax[i % num_slice], fraction=0.05))
                else:
                    cbar[i % num_slice].update_normal(im)
            i += 1
        else:
            break

    plt.plot(sample_time, sample_temp)
    plt.show()

    return latest_mrti[res_idx]

def process_cem(initial_temp, cem_temp, 
                roi, np_folder, num_slice, res_idx, until_idx,
                plot=True):
    if plot:
        fig, ax = plt.subplots(1, num_slice)
        plt.cla()
        plt.subplots_adjust(left=0.01, bottom=0.05, right=0.99, top=0.95, \
                            wspace=0.03, hspace=0.2)

    i = 0
    cbar = []
    sample_pts = []
    cem_imgs = []
    prev_time = []

    with open(np_folder + "/timestamps.txt") as f:
        timestamps = f.readlines()

    while i < until_idx:
        latest_filename = np_folder + '/' + str(i + 1) + ".npy"
        if exists(latest_filename):
            latest_npy = np.load(latest_filename)
            if not roi:
                roi = [0, 0, latest_npy.shape[0], latest_npy.shape[1]]
            latest_mrti = latest_npy[roi[0]:roi[2], roi[1]:roi[3]]
            if len(cem_imgs) < num_slice:
                latest_cem = np.zeros_like(latest_mrti)
                cem_imgs.append(np.zeros_like(latest_mrti))
                cur_time = 0
                prev_time.append(0) 
            else:
                latest_mrti += initial_temp
                cur_time = float(timestamps[i].split(',')[1][:-1])
                latest_cem = update_cem(base_temp=cem_temp,
                                        cem_map=cem_imgs[i % num_slice], 
                                        thermal_img=latest_mrti, 
                                        time_step=cur_time-prev_time[i % num_slice])
                cem_imgs[i % num_slice] = copy.deepcopy(latest_cem)
                prev_time[i % num_slice] = cur_time

            if plot:
                im = ax[i % num_slice].imshow(latest_cem, vmin=0, vmax=30)

            if plot:
                ax[i % num_slice].set_title("Slicer:" + str(i % num_slice + 1) + \
                                            " Time: " + str(cur_time))
                ax[i % num_slice].axis('off')

                if i < num_slice:
                    cbar.append(plt.colorbar(im, ax=ax[i % num_slice], fraction=0.05))
                else:
                    cbar[i % num_slice].update_normal(im)
                plt.pause(0.3)
            i += 1
    
    return cem_imgs[res_idx]

def crop_and_padding(input_img, probe_center, roi, result_size):
    cropped_img = input_img[roi[1]:roi[3], roi[0]:roi[2]]
    print(f"Cropped shape: {cropped_img.shape}")

    x_probe, y_probe = probe_center[0], probe_center[1]
    roi_a_x, roi_a_y = roi[0], roi[1]
    roi_b_x, roi_b_y = roi[2], roi[3]

    padding_left = 50 - (x_probe - roi_a_x) 
    padding_right = 50 - (roi_b_x - x_probe)
    padding_up = 50 - (y_probe - roi_a_y)
    padding_down = 50 - (roi_b_y - y_probe)
    print(f"Padding: left {padding_left}, right {padding_right}, \n \
                    up {padding_up}, down {padding_down}.")

    padded_img = np.pad(cropped_img, 
                        pad_width=((padding_up, padding_down), (padding_left, padding_right)))
    
    return padded_img