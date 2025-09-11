import logging
import os
import pickle
import threading

import cv2
import matplotlib.pyplot as plt
import numpy as np
import yaml
from matplotlib.collections import PatchCollection
from matplotlib.patches import Polygon
from matplotlib.path import Path
from matplotlib.widgets import PolygonSelector
from pydicom import dcmread

from compute_prfs import compute_prfs
from dicom_buffer import DicomBuffer
from mrti_utils import init_plot, update_cem
from viz_npy import update_mrti_plot


def load_mrti_config(mrti_config_path):
    if not os.path.exists(mrti_config_path):
        raise FileNotFoundError(f"Configuration file not found: {mrti_config_path}")
    
    with open(mrti_config_path, 'r') as f:
        mrti_config = yaml.safe_load(f)

    mrti_config["RESULT_FOLDER"] = f"{mrti_config['DICOM_FOLDER']}-results"

    # # Get the first dicom filename under the dicom folder
    # mrti_config["FIRST_DICOM_FILENAME"] = os.listdir(mrti_config["DICOM_FOLDER"])[0] 

    first_dicom_name = mrti_config["FIRST_DICOM_FILENAME"].replace('.dcm', '')
    filename_parts = first_dicom_name.split('.')
    
    if len(filename_parts) < 3:
        raise ValueError(f"Invalid DICOM name format: {first_dicom_name}")

    mrti_config["START_IDX"] = int(filename_parts[-1])
    mrti_config["DICOM_PREFIX"] = '.'.join(filename_parts[:2]) + '.'

    mrti_config["FIRST_DICOM_PATH"] = os.path.join( 
                                        mrti_config["DICOM_FOLDER"], 
                                        f"{mrti_config['DICOM_PREFIX']}{mrti_config['START_IDX']}.dcm")

    if not os.path.exists(mrti_config["FIRST_DICOM_PATH"]):
        raise FileNotFoundError(f"First DICOM file not found: {mrti_config['FIRST_DICOM_PATH']}")

    return mrti_config


def get_dicom_header(dicom_path):
    if not os.path.exists(dicom_path):
        raise FileNotFoundError(f"Missing DICOM file: {dicom_path}")
    
    try:
        dicom_file = dcmread(dicom_path)
        pixel_spacing = dicom_file.PixelSpacing
        pixel_w, pixel_h = pixel_spacing[0], pixel_spacing[1]

        desired_pixel_size = 0.7
        scaling_factor = 143 * 0.7 * 0.5

        half_desired_height_pixel = round(scaling_factor / pixel_h)
        half_desired_width_pixel = round(scaling_factor / pixel_w)

        dicom_header_info = {
            "Height": dicom_file.Rows,
            "Width": dicom_file.Columns,
            "AcquisitionTime": int(dicom_file.AcquisitionTime),
            "MagneticFieldStrength": int(dicom_file.MagneticFieldStrength),
            "EchoTime": int(dicom_file.EchoTime),
            "halfDesiredHeightPixel": half_desired_height_pixel,
            "halfDesiredWidthPixel": half_desired_width_pixel,
            "imageScaledPixelHeight": round(half_desired_height_pixel * 2 * pixel_h / desired_pixel_size),
            "imageScaledPixelWidth": round(half_desired_width_pixel * 2 * pixel_w / desired_pixel_size)}

    except Exception as e:
        logging.error(f"Error reading DICOM file: {str(e)}")
        raise

    return dicom_header_info


def draw_roi(sample_img, prfs_roi_filename):
    # Set up the figure and axes
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 5))
    fig.suptitle("Select the ROI for PRFS; Press any key to save and quit.")
    ax1.imshow(sample_img)
    ax1.set_title("Outer ROI")
    ax2.imshow(sample_img)
    ax2.set_title("Inner ROI")

    # Initialize the ROI mask
    roi = np.zeros(sample_img.shape[:2], dtype=np.uint8)
    
    # Define callback functions for ROI selections
    def onselect_outer(verts):
        logging.info("Outer ROI selected.")
        path = Path(verts)
        vertices = np.array(path.vertices, dtype=int)
        cv2.fillPoly(roi, [vertices], 1)
        p1 = Polygon(vertices, color=[1, 1, 0], alpha=0.3)
        p = PatchCollection([p1], match_original=True)
        ax1.add_collection(p)
        ax1.figure.canvas.draw_idle()

    def onselect_inner(verts):
        logging.info("Inner ROI selected.")
        path = Path(verts)
        vertices = np.array(path.vertices, dtype=int)
        cv2.fillPoly(roi, [vertices], 0)
        p1 = Polygon(vertices, color=[1, 0, 0], alpha=0.3)
        p = PatchCollection([p1], match_original=True)
        ax2.add_collection(p)
        ax2.figure.canvas.draw_idle()

    # Disconnect all events and save the ROI
    def finalize(event):
        fig.canvas.mpl_disconnect(cid_outer)
        fig.canvas.mpl_disconnect(cid_inner)
        np.save(prfs_roi_filename, roi)
        logging.info(f"PRFS ROI saved to {prfs_roi_filename}")
        plt.close(fig)

    # Set up PolygonSelectors with updated props parameter
    poly1 = PolygonSelector(ax1, onselect_outer, useblit=True, props=dict(color='yellow', linestyle='-', linewidth=2, alpha=0.5))
    poly2 = PolygonSelector(ax2, onselect_inner, useblit=True, props=dict(color='red', linestyle='-', linewidth=2, alpha=0.5))

    # Connect 'finalize' to key press event to save ROI when done
    cid_outer = fig.canvas.mpl_connect('key_press_event', finalize)
    cid_inner = fig.canvas.mpl_connect('key_press_event', finalize)
    
    plt.show()

    return roi


def get_dicom_shell(self):
        start_idx = self.mrti_config["START_IDX"]
        slice_num = self.mrti_config["SLICE_NUM"]
        img_per_scan = self.mrti_config["IMG_PER_SCAN"]

        dicom_shell = []
        dicom_set = {}
        for i in range(start_idx, start_idx + img_per_scan):
            filename = os.path.join(self.mrti_config["DICOM_FOLDER"], f'{self.mrti_config["DICOM_PREFIX"]}{i}.dcm')
            dicom_file = dcmread(filename, specific_tags=None)
            i_num = dicom_file.InstanceNumber
            dicom_set[i_num] = dicom_file

        sorted_dicom_dict = dict(sorted(dicom_set.items()))
        sorted_dicom_set = list(sorted_dicom_dict.values())

        for i in range(slice_num):
            dicom_shell.append(sorted_dicom_set[int(i * (img_per_scan / slice_num))])
            print("Slice " + str(i))
            # print(dicom_shell[-1].InstanceNumber)
            print(dicom_shell[-1].ImagePositionPatient)

        return dicom_shell

def setup_igtl(self):
    port = self.mrti_config["IGTL_PORT"]
    igtl_lib = self.mrti_config['IGTL_LIB'].lower()

    if igtl_lib == 'pyigtl':
        from igtl_utlis import launch_pyigtl_server
        return launch_pyigtl_server(port), None
    elif igtl_lib == 'openigtlink':
        from dicomToigtl import export_imgmsg
        from igtl_utlis import launch_openigtlink_client
        socket = launch_openigtlink_client(port)
        dicom_shell = self.get_dicom_shell()
        return socket, dicom_shell
    else:
        raise ValueError(f"Invalid IGTL library specification: {igtl_lib}")


class MRTI:
    def __init__(self, mrti_config):
        self.mrti_config = mrti_config

        # Create result folder
        self.result_folder = mrti_config["RESULT_FOLDER"]
        os.makedirs(self.result_folder, exist_ok=True)

        # Get DICOM header 
        dicom_info_filename = os.path.join(mrti_config["RESULT_FOLDER"], "dicom_info.pkl")
        if os.path.exists(dicom_info_filename):
            logging.info(f"Loading DICOM info from {dicom_info_filename}")
            with open(dicom_info_filename, "rb") as dicom_info_file:
                self.dicom_header_info = pickle.load(dicom_info_file)
        else:
            logging.info("DICOM info file not found. Reading from the first DICOM file.")
            self.dicom_header_info = get_dicom_header(mrti_config["FIRST_DICOM_PATH"])

            with open(dicom_info_filename, "wb") as f:
                pickle.dump(self.dicom_header_info, f)

        # Get PRFS info
        prfs_roi_filename = os.path.join(mrti_config["RESULT_FOLDER"], "prfs_roi.npy")
        if os.path.exists(prfs_roi_filename):
            logging.info(f"Loading PRFS ROI from {prfs_roi_filename}")
            self.prfs_roi = np.load(prfs_roi_filename)
        else:
            logging.info("PRFS ROI not found. Initiating ROI drawing process.")
            ds = dcmread(mrti_config["FIRST_DICOM_PATH"])
            self.prfs_roi = draw_roi(ds.pixel_array, prfs_roi_filename)

        self.prfs_constants = {"rhrcctrlfactor": 3,
                            "alpha": -0.01e-6,
                            "gamma": 42.58e6,
                            # "alpha": -9.09e-9,
                            # "gamma": 268000000,
                            "number_of_coils": 2,
                            "numberslices": mrti_config["SLICE_NUM"],
                            "tempScaleLower": -1,
                            "tempScaleUpper": 1}

        # Set up IGTL communication module
        self.igtl_enabled = mrti_config["IGTL_ENABLED"]
        if self.igtl_enabled:
            self.socket, self.dicom_shell = self.setup_igtl()

        # Set up MRTI configuration
        self.img_per_scan = mrti_config["IMG_PER_SCAN"]
        self.loaded_img_per_scan = int(self.img_per_scan / 3 * 2)
        self.slice_num = mrti_config["SLICE_NUM"]
        self.initial_temp = mrti_config["INIT_TEMP"]
        self.slice_spacing = mrti_config["SLICE_SPACING"]
        self.temp_range = (self.initial_temp, mrti_config["MAX_TEMP"])

        # Latest data
        self.temp_imgs = [None] * self.slice_num
        self.cem_imgs = [None] * self.slice_num
        self.timestamp = 0

        # Set up plot
        self.plot_enabled = mrti_config["PLOT_ENABLED"]
        if self.plot_enabled:
            self.fig, self.ax, self.cbar = init_plot(row=2, col=self.slice_num)

        # Set up dicom file monitoring
        self.watcher = DicomBuffer(debug=False)
        self.lock = threading.Lock()
        self.watcher_thread = threading.Thread(target=self.watcher.start_monitoring, args=(mrti_config, self.lock))
        self.watcher_thread.start()

        self.mrti_stop_flag = False
    
    def get_image_position_patient(self):
        if self.watcher.image_position_patient_acquired:
            return self.watcher.image_position_patient
        else:
            return None
        
    def stop_mrti_loop(self):
        self.watcher.stop_monitoring()
        self.watcher_thread.join()

        print("Stopping MRTI loop...")
        self.mrti_stop_flag = True

    def mrti_loop(self):
        counter, start_time = 0, 0
        baseline, cem_imgs = [], []
        cur_timestamp, prev_timestamp = 0, 0


        while not self.mrti_stop_flag:
            if len(self.watcher.dicom_set) > 0 and len(self.watcher.dicom_set) % self.loaded_img_per_scan == 0:
                with self.lock:
                    self.watcher.hold_flag = True
                    sorted_dicom_set = dict(sorted(self.watcher.dicom_set.items()))
                    current_img_set = list(sorted_dicom_set.values())
                    cur_timestamp = self.watcher.cur_timestamp

                    for i in range(self.slice_num):
                        one_slice_one_scan = self.get_slice_scan(current_img_set, i)
                        idx = counter % self.slice_num
                        del_t_img, uncorrected, baseline_ = self.process_prfs(one_slice_one_scan, baseline, idx)

                        if len(baseline) < self.slice_num:
                            baseline.append(baseline_)
                            self.cem_imgs[idx] = np.zeros_like(del_t_img)
                            if len(baseline) == self.slice_num:
                                start_time = int(cur_timestamp)
                        else:
                            thermal_img = del_t_img + self.initial_temp
                            self.temp_imgs[idx] = thermal_img
                            self.timestamp = int(cur_timestamp)
                            time_step = int(cur_timestamp) - int(prev_timestamp)
                            self.cem_imgs[idx] = update_cem(self.mrti_config["CEM_REF_TEMP"], self.cem_imgs[idx], thermal_img, time_step)
                            
                            if self.plot_enabled:
                                update_mrti_plot(self.ax, self.cbar, thermal_img, self.cem_imgs, idx, int(cur_timestamp) - start_time, self.temp_range, None)
                        self.save_result(counter, del_t_img)
                        
                        if self.igtl_enabled:
                            send_igtl_messages(socket, dicom_shell[i], del_t_img, self.cem_imgs[idx], i, self.slice_spacing)

                        counter += 1

                    prev_timestamp = cur_timestamp
                    self.watcher.dicom_set.clear()
                    self.watcher.hold_flag = False
                    print("Processing done for one scan. Started file monitoring.\n")

    def get_slice_scan(self, current_img_set, i):
        if self.loaded_img_per_scan / self.slice_num == 9:
            return current_img_set[i * 9: i * 9 + 9]  # WPI Dual-channel
        elif self.loaded_img_per_scan / self.slice_num == 2:
            return current_img_set[i * 2: i * 2 + 2]  # GE or WPI Single-channel
        else:
            raise ValueError("Invalid number of images per scan")

    def process_prfs(self, one_slice_one_scan, baseline, idx):
        if len(baseline) < self.slice_num:
            return compute_prfs(self.prfs_constants, self.prfs_roi, self.dicom_header_info, one_slice_one_scan, baseline=None, debug=False)
        else:
            return compute_prfs(self.prfs_constants, self.prfs_roi, self.dicom_header_info, one_slice_one_scan, baseline[idx], debug=False)

    def save_result(self, counter, del_t_img):
        result_filename = os.path.join(self.result_folder, f"{counter:04d}.npy")
        np.save(result_filename, del_t_img)


# def send_igtl_messages(socket, dicom_shell, del_t_img, cem_img, i, slice_spacing):
#     for image, device_name in [(del_t_img, f"MRTI_TEMP_{i}"), (cem_img, f"MRTI_CEM_{i}")]:
#         img_msg = export_imgmsg(image, dicom_shell, device_name, slice_spacing)
#         socket.Send(img_msg.GetPackPointer(), img_msg.GetPackSize())


def main():
    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

    try:
        # Load MRTI configuration
        mrti_config_path = "./mrti_config.yml"       
        mrti_config = load_mrti_config(mrti_config_path)

        mrti_obj = MRTI(mrti_config)

        # Start MRTI loop
        mrti_obj.mrti_loop()

    except Exception as e:
        logging.error(f"An error occurred: {str(e)}", exc_info=True)
        raise

if __name__ == "__main__":
    main()
