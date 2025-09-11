import glob
import os
import random
from datetime import datetime

import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
import numpy as np
from pydicom import dcmread

from constants import FIG_DIR, LESION_MASK_DIR


class LesionPattern:
    def __init__(self, dataset):
        if dataset == "test":
            self.tumor_segs = sorted(glob.glob(LESION_MASK_DIR + '/test/*.npy'), key=os.path.getmtime)
        elif dataset == "train":
            self.tumor_segs = glob.glob(LESION_MASK_DIR + '/train/*.npy')
        elif dataset == "icap":
            self.tumor_segs = glob.glob(LESION_MASK_DIR + '/icap/*.npy')
            print(self.tumor_segs)
        elif dataset == "demo":
            self.tumor_segs = ["/home/yiwei/ConformalAblation/lesion_masks/test/000014-001-S50.npy"]

    def masking(self, seg_idx=None):
        # Lesion Mask 0 -> healthy tissue, 1 -> tumor
        if seg_idx is None:
            seg_idx = random.randint(0, len(self.tumor_segs) - 1)
        seg_fname = self.tumor_segs[seg_idx]
        lesion_mask = np.load(seg_fname)
        num_lesion_pixel = np.count_nonzero(lesion_mask==1)
        
        return lesion_mask, num_lesion_pixel, seg_fname

    def circle_masking(self):
        center = (72, 72)
        radius = 40
        for i in range(self.lesion_mask.shape[0]):
            for j in range(self.lesion_mask.shape[1]):
                if np.sqrt(pow(np.abs(i - center[0]), 2) + pow(np.abs(j - center[1]), 2)) < radius:
                    self.lesion_mask[i, j] = 1
    
    def display_lesion(self):
        row, col = 2, 2
        fig, ax = plt.subplots(row, col)
        
        # tumor_segs = glob.glob(LESION_MASK_DIR + '/*.npy')
        tumor_segs = [r"C:\Users\adam\Sound\ConformalAblation\lesion_masks\test\00000-000-S73.npy",
                      r"C:\Users\adam\Sound\ConformalAblation\lesion_masks\test\00002-000-S75.npy",
                        r"C:\Users\adam\Sound\ConformalAblation\lesion_masks\test\00003-000-S114.npy",
                        r"C:\Users\adam\Sound\ConformalAblation\lesion_masks\test\00005-000-S91.npy",
                        ]
        
        for i in range(len(tumor_segs)):
            self.lesion_mask = np.load(tumor_segs[i])

            ax[int(i / col)][i % col].imshow(self.lesion_mask)
            ax[int(i / col)][i % col].axis('off')
            # print(f"Load lesion mask: {i}.")
            plt.imshow(self.lesion_mask)
        plt.subplots_adjust(left=0.02, bottom=0.1, right=0.98, top=0.9, wspace=0.0, hspace=0.1)
        plt.savefig(FIG_DIR + f"/{datetime.now().strftime('%Y%m%d-%H-%M-%S')}.png")
        # plt.show()
    
    def display_original_seg(self, filename):
        dicom_file = dcmread(filename, specific_tags=None)
        img = dicom_file.pixel_array

        values = np.unique(img.ravel())

        im = plt.imshow(img)

        colors = [ im.cmap(im.norm(value)) for value in values]
        # NCR/NET: necrotic and non-enhancing tumor core
        # ED: peritumoral edema
        # ET: Gadolinium-enhancing tumor
        labels = ["Healthy", "NCR/NET", "ED", "ET"]
        patches = [ mpatches.Patch(color=colors[i], label=labels[i]) for i in range(len(values))]

        plt.legend(handles=patches, loc=0)
        plt.axis('off')
        plt.show()


if __name__ == "__main__":
    lesion_pattern = LesionPattern("icap")
    lesion_pattern.display_lesion()

    # lesion_pattern.display_original_seg(r"/home/yiwei/ConformalAblation/brast2020/2-S49.dcm")

