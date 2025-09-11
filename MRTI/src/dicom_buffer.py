import os
import time
from os.path import exists

from pydicom import dcmread


class DicomBuffer:
    def __init__(self, debug):
        self.dicom_set = {}
        self.hold_flag = False
        self.debug = debug
        self.cur_timestamp = 0
        self.image_position_patient = []
        self.image_position_patient_acquired = False
        self.stop_monitoring_flag = False

    def stop_monitoring(self):
        print("Stopping DICOM monitoring...")
        self.stop_monitoring_flag = True

    def start_monitoring(self, mrti_config, lock):
        cur = last = mrti_config["START_IDX"]
        print(mrti_config["START_IDX"])
        while not self.stop_monitoring_flag:
            cur_filename = os.path.join(mrti_config["DICOM_FOLDER"], f'{mrti_config["DICOM_PREFIX"]}{cur}.dcm')
            if self.debug:
                print("Looking for " + cur_filename)
            
            if cur > last and (cur - last) % mrti_config["IMG_PER_SCAN"] == 0:
                last = cur
                self.image_position_patient_acquired = True
                self.hold_flag = True
                print("Got a dicom set for one scan, hold file monitoring, waiting for processing.\n")
            
            if exists(cur_filename) and not self.hold_flag:
                # Only load the 2nd and 3rd image in each scan (index 1 and 2)
                if (cur - last) % 3 in [1, 2]:
                    dicom_file = dcmread(cur_filename, specific_tags=None)
                    i_num = int(dicom_file.InstanceNumber)

                    self.cur_timestamp = dicom_file.AcquisitionTime
                    if not self.image_position_patient_acquired and ((cur - last) % 3) == 1:
                        # print(cur_filename)
                        self.image_position_patient.append(dicom_file.ImagePositionPatient)
                        # print(self.image_position_patient)
                        
                    lock.acquire()
                    self.dicom_set[i_num] = dicom_file.pixel_array
                    lock.release()
                cur += 1# Add a small delay to prevent busy-waiting
