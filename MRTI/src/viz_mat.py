import glob
import pickle

import matplotlib.pyplot as plt
import numpy as np
import scipy.io
from pydicom import dcmread

mat = scipy.io.loadmat('/mnt/sda1/data/MRTI/mat_data/Tmaps.mat')
pass
ds = np.array(mat['correctedPRFS'])
np.save("res.npy", ds)

# mrti = ds[:, :, 4, 2]
# print(mrti[125, 132])
#
# plt.imshow(mrti)
# plt.show()
# #
# for i in range(26):
#     mrti = ds[:, :, 4, i]
#     print(mrti[125, 132])
#     plt.cla()
#     plt.imshow(mrti)
#     # plt.imshow(self.celsius_images[:, :, i], vmin=20, vmax=50)
#     # plt.text(0.5, 0.5, str(i))
#     plt.pause(3)
    
# mat = scipy.io.loadmat('/mnt/sda1/data/MRTI/mat_data/correctionmaskROI.mat')
# roi = np.array(mat['corROI'])
# prfs_roi_filename = "/mnt/sda1/data/MRTI/mat_data/test-results/" + "prfs_roi.npy"
# np.save(prfs_roi_filename, roi)

# dicom_info_filename = "/mnt/sda1/data/MRTI/mat_data/test-results/" + "dicom_info.plk"
# sample_dicom_file = dcmread('/mnt/sda1/data/MRTI/mat_data/plane5/5.dcm')
# pixel_spacing = sample_dicom_file.PixelSpacing
# pixel_w, pixel_h = pixel_spacing[0], pixel_spacing[1]

# desiredPixelHeight = 0.7
# desiredPixelWidth = 0.7
# dicom_scaling_struct = {"halfDesiredHeightPixel": round(143 * 0.7 * 0.5 / pixel_h),
#                         "halfDesiredWidthPixel": round(143 * 0.7 * 0.5 / pixel_w)}
# dicom_scaling_struct["imageScaledPixelHeight"] = round(
#     dicom_scaling_struct["halfDesiredHeightPixel"] * 2 * pixel_h /
#     desiredPixelHeight)
# dicom_scaling_struct["imageScaledPixelWidth"] = round(
#     dicom_scaling_struct["halfDesiredWidthPixel"] * 2 * pixel_w /
#     desiredPixelWidth)

# dicom_header_info = {"Height": sample_dicom_file.Rows,
#                         "Width": sample_dicom_file.Columns,
#                         "AcquisitionTime": int(sample_dicom_file.AcquisitionTime),
#                         "MagneticFieldStrength": int(sample_dicom_file.MagneticFieldStrength),
#                         "EchoTime": int(sample_dicom_file.EchoTime)}



# f = open(dicom_info_filename,"wb")
# pickle.dump(dicom_header_info, f)
# f.close()

