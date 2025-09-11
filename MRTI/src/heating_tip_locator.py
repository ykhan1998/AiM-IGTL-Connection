import numpy as np
from main import MRTI, load_mrti_config
import threading
import time
import cv2
import os
from collections import deque
import sys

class HeatingTipLocator:
    def __init__(self, prfs_roi):
        # prfs_roi is a binary image with 0 and 1
        # Outer background is 0, outer tissue ROI is 1, inner heating ROI is 0. 
        binary_image = prfs_roi

        # Find the outer black border (connected to the image edges)
        # Use floodFill starting from the corners to fill the outer black region with a unique value
        h, w = binary_image.shape
        mask = np.zeros((h + 2, w + 2), np.uint8)

        # Flood fill the border areas with value 2, starting from each corner, 
        cv2.floodFill(binary_image, mask, (0, 0), 2)
        cv2.floodFill(binary_image, mask, (0, h - 1), 2)
        cv2.floodFill(binary_image, mask, (w - 1, 0), 2)
        cv2.floodFill(binary_image, mask, (w - 1, h - 1), 2)

        # Convert the outer filled area (value 2) to 1
        binary_image[binary_image == 2] = 1

        # Reverse the binary image, so that the heating ROI is 1 and the rest is 0
        self.heating_roi = 1 - binary_image

        self.count = 0

    def locate_heating_tip(self, temp_imgs):
        largest_contour_area = []
        for idx, temp_img in enumerate(temp_imgs):
            contour_area = self.process_image(temp_img, idx, gaussian_blur_kernel_size=(3, 3), threshold_value=4, debug=True)
            largest_contour_area.append(contour_area)
            print(f"Idx: {idx}, Largest Contour area: {contour_area}")

        # Find the index of the image with the largest contour area
        max_contour_idx = np.argmax(largest_contour_area)
        print(f"Max contour index: {max_contour_idx}, Max contour area: {largest_contour_area[max_contour_idx]}")

        return max_contour_idx, largest_contour_area[max_contour_idx]


    def process_image(self, temp_img, idx, gaussian_blur_kernel_size, threshold_value, debug=False):
        '''
        Process the temperature image to find the largest contour area.
        '''
        # Mask the temp_image using the heating ROI
        temp_img = temp_img * self.heating_roi

        # print statistics of the temp_img excluding the background
        print(f"Temp diff min: {np.min(temp_img[temp_img != 0])}, max: {np.max(temp_img[temp_img != 0])}, avg: {np.mean(temp_img[temp_img != 0])}, std: {np.std(temp_img[temp_img != 0])}")

        # Set negative values to 0
        temp_img[temp_img < 0] = 0

        temp_img_uint8 = np.uint8(temp_img)

        blurred = cv2.GaussianBlur(temp_img_uint8, gaussian_blur_kernel_size, 0)

        _, thresh = cv2.threshold(blurred, threshold_value, 255, cv2.THRESH_BINARY)

        # save the threshold image
        if debug:
            cv2.imwrite(f"temp_img_{idx}.png", temp_img_uint8)
            cv2.imwrite(f"blurred_img_{idx}.png", blurred)
            cv2.imwrite(f"thresh_img_{idx}.png", thresh)

        contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        # Filter the largest contour or the one with the hottest spot
        if contours:
            largest_contour = max(contours, key=cv2.contourArea)
            largest_contour_area = cv2.contourArea(largest_contour)
            return largest_contour_area
        else:
            return 0


def load_prfs_roi(mrti_config):
    prfs_roi_filename = os.path.join(mrti_config["RESULT_FOLDER"], "prfs_roi.npy")
    if os.path.exists(prfs_roi_filename):
        print(f"Loading PRFS ROI from {prfs_roi_filename}")
        prfs_roi = np.load(prfs_roi_filename)
        return prfs_roi
    else:
        print(f"PRFS ROI file {prfs_roi_filename} does not exist")
        exit(1)
        

def print_image_position_patient(mrti_obj):
    while True:
        image_position_patient = mrti_obj.get_image_position_patient()
        if image_position_patient:
            for idx in range(len(image_position_patient)):
                print(f"Image {idx} position: {image_position_patient[idx]}")

            break
        time.sleep(0.1)


def detection_loop(mrti_obj, heating_tip_locator):
    prev_timestamp = 0
    idx_queue = deque(maxlen=3)
    area_threshold = 50

    print("Starting heating tip detection...")
    while True:
        temp_imgs = mrti_obj.temp_imgs
        if all(img is not None for img in temp_imgs):
            current_timestamp = mrti_obj.timestamp
            if current_timestamp != prev_timestamp:
                prev_timestamp = current_timestamp
                idx, contour_area = heating_tip_locator.locate_heating_tip(temp_imgs)

                if contour_area > area_threshold:
                    idx_queue.append(idx)
                    if len(idx_queue) == idx_queue.maxlen:
                        if all(i == idx_queue[0] for i in idx_queue):
                            print(f"Heating tip found in image {idx}")
                            break
        time.sleep(0.1)

    return 


if __name__ == "__main__":
    mrti_config_path = "./mrti_config.yml"       
    mrti_config = load_mrti_config(mrti_config_path)

    # Load the PRFS ROI
    prfs_roi = load_prfs_roi(mrti_config)

    # Initialize the heating tip locator
    heating_tip_locator = HeatingTipLocator(prfs_roi)

    # Start MRTI loop in a separate thread
    mrti_obj = MRTI(mrti_config)
    mrti_thread = threading.Thread(target=mrti_obj.mrti_loop)
    mrti_thread.start()

    print_image_position_patient(mrti_obj)

    detection_loop(mrti_obj, heating_tip_locator)

    # Signal the MRTI thread to stop
    mrti_obj.stop_mrti_loop()
    
    # Wait for the MRTI thread to finish
    mrti_thread.join()

    print("All threads have finished. Exiting.")
    sys.exit(0)


