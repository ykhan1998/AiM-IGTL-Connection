import cv2
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import numpy as np
import scipy.io


def fmt(x, pos):
    a, b = '{:.2e}'.format(x).split('e')
    b = int(b)
    return r'${} \times 10^{{{}}}$ Pa'.format(a, b)

def viz_acpr():
    acpr_fname1=r"/home/yiwei/ConformalAblation/ACPR/voltage/P180-10V.mat"
    acpr_fname2=r"/home/yiwei/ConformalAblation/ACPR/voltage/P180-15V.mat"
    acpr_fname3=r"/home/yiwei/ConformalAblation/ACPR/voltage/P180-20V.mat"

    a_field1 = scipy.io.loadmat(acpr_fname1)['acoustic_field']
    a_field2 = scipy.io.loadmat(acpr_fname2)['acoustic_field']
    a_field3 = scipy.io.loadmat(acpr_fname3)['acoustic_field']

    a_fields = [a_field1, a_field2, a_field3]
    plots = [None] * 3

    # visulize the 3 acoustic fields
    fig, axs = plt.subplots(1, 3, figsize=(15, 5))
    fig.suptitle('Acoustic Field, 180-degree, 2W, 3W, 4W', y=0.9)
    for i in range(3):
        plots[i] = axs[i].imshow(a_fields[i], vmin=0, vmax=700000, cmap='terrain', extent=[0,100,0,100])
        axs[i].set_xlabel('X (mm)')
        axs[i].set_ylabel('Y (mm)')
        axs[i].set_title(f'{(i+2)}W')

    cb = fig.colorbar(plots[2], ax=axs, orientation='vertical', shrink=0.75, format=ticker.FuncFormatter(fmt))
    plt.subplots_adjust(left=0.1,
                    bottom=0.1,
                    right=0.75,
                    top=0.9,
                    wspace=0.3)
    plt.show()

def scale_traj(traj_fname):
    with open(traj_fname) as f:
        traj = f.readlines()
        scaled_timestamps = []
        angles = []
        for line in traj:
            timestamp = float(line.split(',')[0])
            angle = float(line.split(',')[1])
            scaled_timestamp = timestamp / 424 * 508
            scaled_timestamps.append(scaled_timestamp)
            angles.append(angle)
    #save scaled traj
    scaled_traj_fname = traj_fname[:-4] + "_scaled.txt"
    with open(scaled_traj_fname, 'w') as f:
        for i in range(len(scaled_timestamps)):
            f.write(f"{round(scaled_timestamps[i])},{round(angles[i])}\n")

if __name__ == "__main__":
    # viz_acpr()

    # scale_traj("/home/yiwei/ConformalAblation/iCAP/traj_save/2022-11-10 13:48:03.358815/traj.txt")

    a = np.load('/mnt/sda1/data/conformal_ablation/20221110/test6-results/cem_mask.npy')

    pass