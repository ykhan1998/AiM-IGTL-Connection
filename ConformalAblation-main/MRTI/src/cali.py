
import cv2
import imutils
import matplotlib.patches as patches
import matplotlib.pyplot as plt

from utils import *
from viz_npy import viz_mrti_results


def clicked_event(event):
    global click_counter, probe_center, roi
    if click_counter == 0:
        probe_center = [int(event.xdata), int(event.ydata)]
        print(f"Probe_center: {probe_center}")
        click_counter += 1
    elif click_counter == 1:
        roi.extend([int(event.xdata), int(event.ydata)])
        click_counter += 1
    elif click_counter == 2:
        roi.extend([int(event.xdata), int(event.ydata)])
        print(f"ROI: {roi}")


if __name__ == "__main__":
    results_folder = r"C:\Users\yiwei\Documents\20230126\test2-results"

    res_temp_img = viz_mrti_results(results_folder, num_slice=5, roi=[], 
                                    vmin=-5, vmax=20, 
                                    res_idx=2, pt=(129, 133), 
                                    plot=False)

    global click_counter, probe_center, roi
    click_counter = 0
    roi = []
    fig = plt.figure()
    ax = fig.add_subplot(111)
    ax.imshow(res_temp_img)
    fig.canvas.mpl_connect('button_press_event', clicked_event)
    print("Select the probe center and then the ROI.")
    plt.show()

    rect = patches.Rectangle((roi[0], roi[1]), roi[2] - roi[0], roi[3] - roi[1], 
                            linewidth=1, edgecolor='r', facecolor='none')
    circle = patches.Circle((probe_center[0], probe_center[1]), 2,
                            linewidth=1, edgecolor='r', facecolor='none')
    fig = plt.figure()
    ax = fig.add_subplot(111)
    ax.imshow(res_temp_img)
    ax.add_patch(rect)
    ax.add_patch(circle)
    plt.show()

    raw_cem_img = process_cem(initial_temp=19, 
                                cem_temp=25, 
                                roi=[],
                                np_folder=results_folder,
                                num_slice=5,
                                res_idx=2,
                                until_idx=95,
                                plot=False)
    
    print("raw_cem_img")
    plt.imshow(raw_cem_img, vmin=0, vmax=30)
    plt.show()

    centered_cem_img = crop_and_padding(raw_cem_img, 
                                        probe_center=probe_center,
                                        roi=roi, 
                                        result_size=(100, 100))

    print(f"Shape after padding: {centered_cem_img.shape}")

    thresh = 0.5
    cem_masked = np.zeros_like(centered_cem_img)
    cem_masked[centered_cem_img > thresh] = 1

    print(f"Burnt: {np.count_nonzero(cem_masked==1)}")

    plt.imshow(cem_masked)
    plt.show()

    # Rotate 
    rotated_cem_mask = imutils.rotate(cem_masked, 0)
    
    plt.imshow(rotated_cem_mask)
    plt.show()

    # Flip
    final_cem_mask = cv2.flip(rotated_cem_mask, 1)
    plt.imshow(final_cem_mask)
    plt.show()

    np.save(results_folder + "/cem_mask.npy", final_cem_mask)

