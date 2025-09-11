import copy
import os
from os.path import exists
from random import random

import matplotlib.patches as patches
import matplotlib.pyplot as plt
import numpy as np


class DrawROI:
    def __init__(self):
        self.click_counter = 0
        self.probe_center = []
        self.roi = []

    def clicked_event(self, event):
        # Click Probe Center
        if self.click_counter == 0:
            self.probe_center = [int(event.xdata), int(event.ydata)]
            print(f"Probe_center: {self.probe_center}")
            self.click_counter += 1
        # Click Left Top
        elif self.click_counter == 1:
            self.roi.extend([int(event.xdata), int(event.ydata)])
            self.click_counter += 1
        # Click Right Bottom
        elif self.click_counter == 2:
            self.roi.extend([int(event.xdata), int(event.ydata)])
            print(f"ROI: {self.roi}")

            # Display selected probe center and ROI
            rect = patches.Rectangle((self.roi[0], self.roi[1]), 
                                    self.roi[2] - self.roi[0], self.roi[3] - self.roi[1], 
                                    linewidth=1, edgecolor='r', facecolor='none')
            circle = patches.Circle((self.probe_center[0], self.probe_center[1]), 
                                    2, linewidth=1, edgecolor='r', facecolor='none')
            self.ax.add_patch(rect)
            self.ax.add_patch(circle) 
            plt.show()
            
    def draw_roi(self, temp_img):
        fig = plt.figure()
        self.ax = fig.add_subplot(111)
        self.ax.imshow(temp_img)
        fig.canvas.mpl_connect('button_press_event', self.clicked_event)
        print("Select the probe center and then the ROI.")
        plt.show()


def init_plot(row, col):
    fig, ax = plt.subplots(row, col)
    plt.cla()
    plt.subplots_adjust(left=0.01, bottom=0.05, right=0.99, top=0.95, \
                        wspace=0.09, hspace=0.2)
    if row == 1:
        cbar = [None for _ in range(col)]
        for c in range(col):
            empty_img = ax[c].imshow(np.zeros((10,10)), vmin=0, vmax=10)
            ax[c].axis('off')
            cbar[c] = plt.colorbar(empty_img, ax=ax[c], fraction=0.03)
        plt.pause(0.2)
    else:
        cbar = [[None for _ in range(col)] for _ in range(row)]
        for r in range(row):
            for c in range(col):
                empty_img = ax[r][c].imshow(np.zeros((10,10)), vmin=0, vmax=10)
                ax[r][c].axis('off')
                cbar[r][c] = plt.colorbar(empty_img, ax=ax[r][c], fraction=0.03)
        plt.pause(0.2)
    return fig, ax, cbar

def write_timestamps(results_folder, num_slice):
    i = 0
    time_base = []
    while True:
        latest_filename = results_folder + '/' + str(i + 1) + ".npy"
        if exists(latest_filename):
            if len(time_base) < num_slice:
                time_base.append(os.path.getmtime(latest_filename))
                cur_time = 0
            else:
                cur_time = os.path.getmtime(latest_filename) - time_base[i % num_slice]
            
            with open(results_folder + "/timestamps.txt", 'a') as f:
                    f.write(str(i + 1) + "," + str(round(cur_time, 2)) + "\n")
            i += 1
        else:
            break

def check_timestamp_file(results_folder):
    if not exists(results_folder + "/timestamps.txt"):
            print("No timestamp file was found, generating timestamp file.")
            write_timestamps(results_folder, num_slice=5)
            print("Timestamp file has been generated.")
    else:
        print("Timestamp file exists.")

def update_cem(ref_temp, cem_map, thermal_img, time_step):
    # time in seconds
    # t = time.time()
    for i in range(thermal_img.shape[0]):
        for j in range(thermal_img.shape[1]):
            T = thermal_img[i][j]
            if T >= ref_temp:
                R = 0.5
            else:
                R = 0.25
            cem_map[i][j] += R ** (ref_temp - T) * time_step / 60 # mintues
    # elapsed = time.time() - t
    # print(elapsed)
    return cem_map

def compute_status_map(lesion_mask, ablation_res_map):
    lesion_mask = lesion_mask.astype(int)
    ablation_res_map = ablation_res_map.astype(int)

    # lesion pixel number
    tumor_ablated_idx = (lesion_mask == 1) & (ablation_res_map == 1)
    tumor_unablated_idx = (lesion_mask == 1) & (ablation_res_map == 0)
    healthy_ablated_idx = (lesion_mask == 0) & (ablation_res_map == 1)

    status_map = np.ones_like(ablation_res_map)*2

    status_map[tumor_ablated_idx] = 3
    status_map[tumor_unablated_idx] = 1
    status_map[healthy_ablated_idx] = 0

    return status_map

def process_cem(init_temp, cem_temp, roi, mrti_folder, num_slice, res_idx, until_idx,
                plot=True):
    if plot:
        fig, ax = plt.subplots(1, num_slice)
        plt.cla()
        plt.subplots_adjust(left=0.01, bottom=0.05, right=0.99, top=0.95, \
                            wspace=0.03, hspace=0.2)

    i = 0
    cbar = []
    cem_imgs = []
    prev_time = []

    with open(mrti_folder + "/timestamps.txt") as f:
        timestamps = f.readlines()

    while i < until_idx:
        print(f"Processing {i}th slice.")
        latest_filename = mrti_folder + '/' + str(i + 1) + ".npy"
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
                latest_mrti += init_temp
                cur_time = float(timestamps[i].split(',')[1][:-1])
                latest_cem = update_cem(ref_temp=cem_temp,
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

    padding_left = int(result_size[0] / 2) - (x_probe - roi_a_x) 
    padding_right = int(result_size[0] / 2) - (roi_b_x - x_probe)
    padding_up = int(result_size[1] / 2) - (y_probe - roi_a_y)
    padding_down = int(result_size[1] / 2) - (roi_b_y - y_probe)
    print(f"Padding: left {padding_left}, right {padding_right}, \n \
                    up {padding_up}, down {padding_down}.")

    padded_img = np.pad(cropped_img, 
                        pad_width=((padding_up, padding_down), (padding_left, padding_right)))
    
    return padded_img

def padding(array, xx, yy):
    """
    :param array: numpy array
    :param xx: desired height
    :param yy: desirex width
    :return: padded array
    """

    h = array.shape[0]
    w = array.shape[1]

    a = (xx - h) // 2
    aa = xx - a - h

    b = (yy - w) // 2
    bb = yy - b - w

    return np.pad(array, pad_width=((a, aa), (b, bb)), mode='constant')