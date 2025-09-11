import glob
import os
import time

import matplotlib.pyplot as plt
import numpy as np
import scipy.io
from PIL import Image
from pydicom import dcmread
import random


def complex2phase(r_img, i_img):
    complex_img = r_img + 1j * i_img
    phase_img = np.angle(complex_img)
    return phase_img


def dicom_viewer():
    dicom_folder = '/mnt/sda1/data/prostate/20220502_ProstateTest/Prostate Test Test/Unknown Study/MR 3D Ax T1 BRAVO-2'
    dicom_files = os.listdir(dicom_folder)
    pre = "/MR00"
    # f'{i:05d}'
    start = 50
    num = 45
    row = 5
    col = int(num / row)
    img = []
    
    fig, ax = plt.subplots(row, col)

    for i in range(num):
        number = str((start + i) )
        number = number.zfill(4)
        filename = dicom_folder + pre + number + ".dcm"
        print(filename)
        ds = dcmread(filename)
        img.append(np.array(ds.pixel_array))
        # print(ds.pixel_array[110][110])
        pos = np.array(ds.ImagePositionPatient)
        pos = np.around(pos, decimals=1)
        instance_num = ds.InstanceNumber

        ax[int(i / col)][i % col].imshow(img[i])
        ax[int(i / col)][i % col].axis('off')
        ax[int(i / col)][i % col].set_title(str(number) + " i_num: " + str(instance_num))
        ax[int(i / col)][i % col].text(0, 280, np.array2string(pos))

    # file_list = sorted(glob.glob(dicom_folder + "/*.dcm"))
    # file_list = sorted(glob.glob(dicom_folder + "/*.dcm"), key=os.path.getmtime)
    # for filename in file_list:
    #     ds1 = dcmread(filename)
    #     img.append(np.array(ds1.pixel_array))


    plt.subplots_adjust(left=0.01, bottom=0.05, right=0.99, top=0.95, wspace=0.03, hspace=0.2)

    plt.show()


if __name__ == "__main__":
    dicom_viewer()
    # result_viewer()

    # dicom_file = dcmread("/mnt/sda1/data/MRTI/mat_data/test/3.dcm")
    # arr = dicom_file.pixel_array
    # m1 = np.array(arr)
    # dicom_file = dcmread("/mnt/sda1/data/MRTI/mat_data/test/10.dcm")
    # arr = dicom_file.pixel_array
    # r1 = np.array(arr)
    # dicom_file = dcmread("/mnt/sda1/data/MRTI/mat_data/test/14.dcm")
    # arr = dicom_file.pixel_array
    # i1 = np.array(arr)

    # [w, h] = np.shape(i1)
    # x = random.randint(0, w-1)
    # y = random.randint(0, h-1)
    # map = np.zeros_like(m1)
    # for x in range(w):
    #     for y in range(h):
    #         mag = m1[x, y]
    #         res = np.sqrt(r1[x, y] **2 + i1[x, y] **2)
    #         if mag == 0:
    #             map[x, y] = 1
    #             continue
    #         if np.abs(res-mag)/mag > 0.03:
    #             map[x, y] = 2
    # plt.imshow(map)
    # plt.show()

    # print(m1[x, y])
    # print(r1[x, y])
    # print(i1[x, y])
    # res = np.sqrt(r1[x, y] **2 + i1[x, y] **2)
    # print(res)

    # phase_1 = complex2phase(r_1, i_1)

    # plt.imshow(phase_1)
    # plt.show()

    # dicom_file = dcmread("/mnt/sda1/data/MRTI/mat_data/test/25.dcm")
    # arr = dicom_file.pixel_array
    # r_2 = np.array(arr)
    # dicom_file = dcmread("/mnt/sda1/data/MRTI/mat_data/test/30.dcm")
    # arr = dicom_file.pixel_array
    # i_2 = np.array(arr)

    # phase_2 = complex2phase(r_2, i_2)

    # del_phase = phase_2 - phase_1

    # plt.imshow(phase_2)
    # plt.show()

    # plt.imshow(del_phase)
    # plt.show()
