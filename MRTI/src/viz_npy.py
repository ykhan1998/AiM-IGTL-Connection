from os.path import exists
from random import random

import cv2
import imutils
import matplotlib.patches as patches
import matplotlib.pyplot as plt
import numpy as np
from mrti_utils import (
    DrawROI,
    check_timestamp_file,
    compute_status_map,
    crop_and_padding,
    init_plot,
    padding,
    process_cem,
    update_cem,
)


def update_mrti_plot(ax, cbar, temp_img, cem_img, i, cur_time, temp_range, 
                     show_status=False, cem_thresh=None, lesion_mask=None):
    # Temperature
    temp_plot = ax[0][i].imshow(temp_img, vmin=temp_range[0], vmax=temp_range[1], cmap='terrain')
    # temp_plot = ax[0][i].imshow(temp_img, cmap='terrain')
    ax[0][i].set_title(f"Slicer {i + 1}\n Time {cur_time}\n Idx {i}")
    ax[0][i].axis('off')
    cbar[0][i].update_normal(temp_plot)

    # Status / CEM
    if show_status:
        cem_masked = np.zeros_like(cem_img[i])
        cem_masked[cem_img[i] > cem_thresh] = 1
        status_map = compute_status_map(lesion_mask, ablation_res_map=cem_masked)
        status_plot = ax[1][i].imshow(status_map, cmap='terrain', vmin=1, vmax=3)
    else:
        cem_plot = ax[1][i].imshow(cem_img[i], vmin=0, vmax=20)
    ax[1][i].axis('off')
    plt.pause(0.3)

def viz_mrti_results(results_folder, num_slice, temp_range, res_idx, 
                     init_temp=19, cem_ref_temp=25,
                     roi=None, ctrl_pt=None, rotate_angle=None,
                     plot=False):
    if plot:
        fig, ax, cbar = init_plot(row=2, col=num_slice)

    if ctrl_pt is not None:
        sample_temp = []
        sample_time = []
    with open(results_folder + "/timestamps.txt") as f:
        timestamps = f.readlines()

    i = 0
    mrti = [0] * num_slice
    cem_imgs = []
    prev_time = [0 for _ in range(num_slice)]
    s_ab_ls = [0]
    t_ls =  [0]

    # pad and resize the 100 * 100 (100 mm * 100 mm) lesion mask to 256 * 256 (220 mm * 220 mm)
    lesion_mask = np.load('/home/yiwei/ConformalAblation/lesion_masks/icap/all_tumor.npy')
    lesion_mask_padded = padding(lesion_mask, 220, 220)
    lesion_mask_padded = cv2.resize(lesion_mask_padded, dsize=(256, 256), interpolation=cv2.INTER_NEAREST)

    while True:
        latest_filename = results_folder + '/' + str(i + 1) + ".npy"
        if exists(latest_filename):
            latest_npy = np.load(latest_filename)
            if roi is None:
                mrti[i % num_slice] = latest_npy
            else:
                mrti[i % num_slice] = latest_npy[roi[0]:roi[2], roi[1]:roi[3]]
            
            cur_time = float(timestamps[i].split(',')[1][:-1])

            if len(cem_imgs) < num_slice:
                cem_imgs.append(np.zeros_like(latest_npy, dtype=np.float32))
            else:
                mrti[i % num_slice] += init_temp
                latest_cem = update_cem(ref_temp=cem_ref_temp,
                                        cem_map=cem_imgs[i % num_slice], 
                                        thermal_img=mrti[i % num_slice], 
                                        time_step=cur_time-prev_time[i % num_slice])
                cem_imgs[i % num_slice] = np.array(latest_cem)
                prev_time[i % num_slice] = cur_time

                ablated_idx = cem_imgs[res_idx][90:170, 80:170] > 1
                s_ab = np.count_nonzero(ablated_idx)
                if round(cur_time) != t_ls[-1]:
                    s_ab_ls.append(s_ab)
                    t_ls.append(round(cur_time))
            
            if rotate_angle is not None:
                mrti[i % num_slice] = imutils.rotate(mrti[i % num_slice], rotate_angle)
                mrti[i % num_slice] = cv2.flip(mrti[i % num_slice], 1)

            if ctrl_pt is not None and i % num_slice == res_idx:
                temp = np.average(mrti[i % num_slice]\
                                  [ctrl_pt[0]-3:ctrl_pt[0]+2, ctrl_pt[1]-3:ctrl_pt[1]+2]) + 19
                sample_temp.append(temp)
                sample_time.append(cur_time)

            if plot:
                idx = i % num_slice
                update_mrti_plot(ax=ax, cbar=cbar, temp_img=mrti[idx], cem_img=cem_imgs,
                                i=idx, cur_time=cur_time, temp_range=temp_range, 
                                show_status=True, cem_thresh=1,
                                lesion_mask=lesion_mask_padded)
            i += 1
            print(f"time: {cur_time}")
        else:
            break

    data = {"sample_temp": sample_time,
            "sample_time": sample_temp,
            "s_ab": s_ab_ls,
            "t_ls": t_ls}

    return mrti[res_idx], data

def viz_ablated_map(res_temp_img, until_idx, cem_thresh, rotate_angle, probe_center=None, roi=None):
    # 256 * 0.8594 mm = 220 * 1 mm
    res_temp_img = cv2.resize(res_temp_img, dsize=(220, 220), interpolation=cv2.INTER_LINEAR)
    draw_roi_obj = DrawROI()
    if probe_center is not None and roi is not None:
        draw_roi_obj.probe_center = probe_center
        draw_roi_obj.roi = roi
    else:
        draw_roi_obj.draw_roi(res_temp_img)

    raw_cem_img = process_cem(init_temp=19, cem_temp=25, 
                                roi=[],
                                mrti_folder=results_folder,
                                num_slice=5,
                                res_idx=2,
                                until_idx=until_idx,
                                plot=False)
    # 256 * 0.8594 mm = 220 * 1 mm
    raw_cem_img = cv2.resize(raw_cem_img, dsize=(220, 220), interpolation=cv2.INTER_LINEAR)

    plt.imshow(raw_cem_img)
    plt.title("Raw CEM")
    plt.show()

    if probe_center is not None:
        draw_roi_obj.probe_center = probe_center

    centered_cem_img = crop_and_padding(raw_cem_img, 
                                        probe_center=draw_roi_obj.probe_center,
                                        roi=draw_roi_obj.roi, 
                                        result_size=(100, 100))
    
    # set all the values greater than 200 to 0
    centered_cem_img[centered_cem_img > 300] = 0

    histo, range_text = [], []
    # calculate the pixel number of the value less than 1
    histo.append(len(centered_cem_img[(centered_cem_img > 0.01) & (centered_cem_img < 0.1)]))
    range_text.append("0.01-0.1")

    histo.append(len(centered_cem_img[(centered_cem_img > 0.1) & (centered_cem_img < 0.5)]))
    range_text.append("0.1-0.5")

    histo.append(len(centered_cem_img[(centered_cem_img > 0.5) & (centered_cem_img < 1)]))
    range_text.append("0.5-1")
    
    histo.append(len(centered_cem_img[(centered_cem_img > 1) & (centered_cem_img < 2)]))
    range_text.append("1-2")

    histo.append(len(centered_cem_img[(centered_cem_img > 2) & (centered_cem_img < 5)]))
    range_text.append("2-5")

    histo.append(len(centered_cem_img[(centered_cem_img > 5) & (centered_cem_img < 10)]))
    range_text.append("5-10")

    histo.append(len(centered_cem_img[(centered_cem_img > 10) & (centered_cem_img < 50)]))
    range_text.append("10-50")


    histo.append(len(centered_cem_img[(centered_cem_img > 50)]))
    range_text.append(">50")

    plt.bar(range_text, histo)
    plt.ylim(0, 800)

    plt.title("Histogram of CEM")
    plt.show()
    plt.savefig("/home/yiwei/ConformalAblation/fig" + "/cem_hist.png")

    # Thresholding
    cem_masked = np.zeros_like(centered_cem_img)
    cem_masked[centered_cem_img > cem_thresh] = 1
    print(f"Burnt: {np.count_nonzero(cem_masked==1)}")
    plt.imshow(cem_masked)
    plt.title("Threshed CEM")
    plt.show()

    # Rotate counter-clockwise
    rotated_cem_mask = imutils.rotate(cem_masked, rotate_angle)
    plt.imshow(rotated_cem_mask)
    plt.title("Rotated CEM")
    plt.show()

    # Flip left-right
    final_cem_mask = cv2.flip(rotated_cem_mask, 1)
    plt.imshow(final_cem_mask)
    plt.title("Flipped CEM")
    plt.show()

    np.save(results_folder + "/ablated_mask.npy", final_cem_mask)

def plot_temp_curve(sample_temp, sample_time):
    plt.plot(sample_time, sample_temp)
    plt.xlabel("Time (s)")
    plt.ylabel("Temperature (C)")
    plt.savefig('/home/yiwei/ConformalAblation/fig' + "/temp_curve.png")

def plot_s_ab_curve(s_ab_ls, t_ls):
    # add 10 to the time curve
    t_ls = [t + 10 for t in t_ls]
    plt.plot(t_ls, s_ab_ls)
    plt.scatter(t_ls, s_ab_ls, s=12)
    plt.xlabel("Time (s)")
    plt.ylabel("Ablated Area (mm^2)")
    plt.savefig('/home/yiwei/ConformalAblation/fig' + "/s_ab_curve.png")

    # t_ls2 = np.load("/home/yiwei/ConformalAblation/iCAP/traj_save/20231030-21-21-02/time_curve.npy")
    # s_ab_ls2 = np.load("/home/yiwei/ConformalAblation/iCAP/traj_save/20231030-21-21-02/cem_curve.npy")
    t_ls2 = np.load("/home/yiwei/ConformalAblation/iCAP/traj_save/20231116-23-35-15/time_curve.npy")
    s_ab_ls2 = np.load("/home/yiwei/ConformalAblation/iCAP/traj_save/20231116-23-35-15/cem_curve.npy")
    
    s_ab_ls2_down = np.array([s_ab_ls2[i] for i in t_ls])

    print(s_ab_ls)
    print(s_ab_ls2_down)
    x = 10
    plt.scatter(t_ls2[:-x], s_ab_ls2[:-x], s=0.5)
    plt.plot(t_ls2[:-x], s_ab_ls2[:-x])
    plt.xlim(0, 160)
    plt.xlabel("Time (s)")
    plt.ylabel("Ablated Area (mm^2)")
    plt.savefig('/home/yiwei/ConformalAblation/fig' + "/s_ab_curve_.png")

    # fit a linear regression of t_ls and s_ab_ls
    from scipy import stats
    slope, intercept, r_value, p_value, std_err = stats.linregress(t_ls[7:], s_ab_ls[7:])
    print(f"r_value: {r_value}")
    print(f"p_value: {p_value}")
    print(f"std_err: {std_err}")
    print(f"slope: {slope}")
    print(f"intercept: {intercept}")

    r = np.corrcoef(s_ab_ls, s_ab_ls2_down)
    print(r[0, 1])

if __name__ == "__main__":
    results_folder = r"/mnt/sda1/data/conformal_ablation/20221110/test6-results"
    check_timestamp_file(results_folder)
    
    res_temp_img, data = viz_mrti_results(results_folder, num_slice=5, temp_range=(20, 40), res_idx=1,
                                    roi=None, ctrl_pt=[128, 126], rotate_angle=None, plot=False)
    
    # s_ab_ls = data["s_ab"]
    # t_ls = data["t_ls"]

    # plot_s_ab_curve(s_ab_ls, t_ls)

    viz_ablated_map(res_temp_img, 
                    until_idx=65, 
                    cem_thresh = 2, 
                    rotate_angle=-60, 
                    probe_center=[108, 113], 
                    roi=[80, 80, 135, 135])
