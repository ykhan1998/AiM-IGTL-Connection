import datetime
import os
import random
import sys
import threading
import time

import cv2
import matplotlib.pyplot as plt
import numpy as np
import yaml
from icap_constants import *
from matplotlib import cm
from PIL import Image, ImageQt
from PyQt5 import QtGui, uic
from PyQt5.QtCore import QThread, QTimer
from PyQt5.QtGui import QImage, QPixmap
from PyQt5.QtWidgets import (QApplication, QComboBox, QLabel, QLineEdit,
                             QMainWindow, QPushButton, QSlider, QSpinBox,
                             QStatusBar, QTextEdit)

sys.path.append(CONFORMAL_ABLATION_DIR)

from lesion_pattern import LesionPattern
from matlab_env import MatlabEnv


class ICAP_UI(QMainWindow):
    def __init__(self):
        super(ICAP_UI, self).__init__()

        uic.loadUi(ICAP_DIR + "/iCAP.ui", self)
        
        self.status_bar = self.findChild(QStatusBar, "status_bar")
        self.status_bar.showMessage(f"Ready to start.")

        # Applicator settings.
        self.applicator_combo_box = self.findChild(QComboBox, "applicator_combo_box")
        self.applicator_combo_box.addItem("Planer")
        self.applicator_combo_box.addItem("90")
        self.applicator_combo_box.addItem("180")
        self.applicator_combo_box.addItem("360")
        self.applicator_combo_box.currentTextChanged.connect(\
            self.applicator_combo_box_changed)
        self.applicator_combo_box.setCurrentIndex(1)
        
        self.power_spin_box = self.findChild(QSpinBox, "power_spin_box")
        self.power_spin_box.valueChanged.connect(self.power_spin_box_changed)
        
        # Tissue settings.
        self.tissue_combo_box = self.findChild(QComboBox, "tissue_combo_box")
        self.tissue_combo_box.addItem("Brain")
        self.tissue_combo_box.addItem("Phantom")
        self.tissue_combo_box.currentTextChanged.connect(self.tissue_combo_box_changed)
        self.tissue_combo_box.setCurrentIndex(1)

        self.tstep_spin_box = self.findChild(QSpinBox, "tstep_spin_box")
        self.tstep_spin_box.valueChanged.connect(self.tstep_spin_box_changed)

        self.tumor_shape_lineedit = self.findChild(QLineEdit, "tumor_shape_lineEdit")
        
        # CEM settings.
        self.init_temp_lineedit = self.findChild(QLineEdit, "init_temp_lineEdit")
        self.cem_temp_lineedit = self.findChild(QLineEdit, "cem_temp_lineEdit")
        self.cem_thresh_lineedit = self.findChild(QLineEdit, "cem_thresh_lineEdit")
        self.confirm_button = self.findChild(QPushButton, "confirm_button")
        self.confirm_button.clicked.connect(self.confirm_button_clicked)
        
        self.start_button = self.findChild(QPushButton, "start_button")
        self.start_button.clicked.connect(self.start_button_clicked)
        self.poweroff_button = self.findChild(QPushButton, "power_off_button")
        self.poweroff_button.clicked.connect(self.poweroff_button_clicked)
        self.stop_button = self.findChild(QPushButton, "stop_button")
        self.stop_button.clicked.connect(self.stop_button_clicked)
        self.reset_button = self.findChild(QPushButton, "reset_button")
        self.reset_button.clicked.connect(self.reset_button_clicked)

        self.angle_slider = self.findChild(QSlider, "angle_slider")
        self.angle_slider.valueChanged.connect(self.angle_slicer_value_changed)
        self.angle_label = self.findChild(QLabel, "angle_label")
        self.time_label = self.findChild(QLabel, "time_label")

        self.save_traj_button = self.findChild(QPushButton, "save_traj_button")
        self.save_traj_button.clicked.connect(self.save_traj_button_clicked)
        self.load_traj_button = self.findChild(QPushButton, "load_traj_button")
        self.load_traj_button.clicked.connect(self.load_traj_button_clicked)
        self.traj_filename_edit = self.findChild(QTextEdit, "traj_TextEdit")

        self.img_1 = self.findChild(QLabel, "img_1")
        self.img_2 = self.findChild(QLabel, "img_2")
        self.img_3 = self.findChild(QLabel, "img_3")

        self.env = None
        self.probe_type = self.applicator_combo_box.currentText() # 90, 180, 360
        self.probe_power = None
        self.tissue_type = self.tissue_combo_box.currentText() # brain/phantom
        self.time_step = 1.0 # seconds

        self.init_temp = None
        self.cem_temp = None
        self.cem_thresh = None # mintues

        self.cur_timestamp = 0  # seconds
        self.current_angle = 0
        self.power_switch = True
        self.traj = []
        self.loaded_traj = []
        self.replay_traj = False
        self.total_reward = 0
        self.simu_thread = None
        self.loop_flag = True
        self.ablation_result_img = None

        self.temp_lst = []
        self.cem_lst = []
        self.time_lst = []

        self.datetime_now = None
        self.env_config_fname = None

        self.lesion_pattern_obj = LesionPattern(dataset="icap")

        self.show()

    def applicator_combo_box_changed(self, value):
        print("applicator_combo_box_changed", value)
        self.probe_type = value

    def power_spin_box_changed(self, value):
        print("power_spin_box_changed", value)
        self.probe_power = int(value)

    def tissue_combo_box_changed(self, value):
        print("tissue_combo_box_changed", value)
        self.tissue_type = value

    def tstep_spin_box_changed(self, value):
        print("tstep_spin_box_changed", value)
        self.time_step = float(value)

    def confirm_button_clicked(self):
        self.init_temp = float(self.init_temp_lineedit.text())
        self.cem_temp = float(self.cem_temp_lineedit.text())
        self.cem_thresh = float(self.cem_thresh_lineedit.text())

        seg_idx = int(self.tumor_shape_lineedit.text())

        print(f"confirm_button_clicked, init_temp: {self.init_temp}, \n\
                                        cem_temp: {self.cem_temp}, \n\
                                        cem_thresh: {self.cem_thresh}, \n\
                                        seg_idx: {seg_idx}")

        env_config = {"HEIGHT": 100,
                        "WIDTH": 100,
                        "INIT_TEMP": self.init_temp,
                        "TEMP_MAX": 70,
                        "CEM_TEMP": self.cem_temp,
                        "CEM_THRESH": self.cem_thresh,
                        "PA_MIN": 0,
                        "PA_MAX": 1000000,
                        "PROBE_TYPE": self.probe_type,
                        # "POWER": self.probe_power,
                        "VOLTAGE": 15,
                        "TIME_STEP": self.time_step,
                        "ANGLE_STEP": 5,
                        "MAX_STEP": 5000}

        self.datetime_now = datetime.datetime.now().strftime("%Y%m%d-%H-%M-%S")

        self.save_dir = ICAP_DIR + f"/traj_save/{self.datetime_now}"
        os.makedirs(self.save_dir)

        self.env_config_fname = self.save_dir + f"/env_config.yml"
        with open(self.env_config_fname, 'w') as f:
            yaml.dump(env_config, f)
        
        self.lesion_mask, num_lesion_pixel, seg_fname = self.lesion_pattern_obj.masking(seg_idx)
        
        with open(self.save_dir + "/log.txt", 'a') as f:
            f.write(f"Tumor shape: {seg_fname}. \n")

    def start_button_clicked(self):
        self.status_bar.showMessage(f"Simulation started.")

        assert os.path.isfile(self.env_config_fname)
        env_config = yaml.load(open(self.env_config_fname), Loader=yaml.FullLoader)

        self.env = MatlabEnv(env_config, verbose=False, plot=False, render_mode="rgb_array")
        self.lesion_mask = cv2.resize(self.lesion_mask, 
                                        dsize=(self.env.height, self.env.width),
                                        interpolation=cv2.INTER_NEAREST)
        self.env.reset(self.lesion_mask, "/home/yiwei/ConformalAblation/ACPR/voltage/P90-20V.mat")

        self.cur_timestamp = 0
        self.total_reward = 0

        self.simu_thread = threading.Thread(target=self.simulation_loop)
        self.loop_flag = True
        self.power_switch = True
        self.simu_thread.start()

    def poweroff_button_clicked(self):
        self.status_bar.showMessage(f"Power off at {self.cur_timestamp}.")
        self.power_switch = False
        with open(self.save_dir + "/log.txt", 'a') as f:
            f.write(f"Power off at {self.cur_timestamp}. \n")

    def stop_button_clicked(self):
        self.loop_flag = False
        if not self.replay_traj:
            self.simu_thread.join()

        self.status_bar.showMessage(f"Simulation stopped at {self.cur_timestamp}." + \
                                    " Total reward: " + str(self.total_reward))
        with open(self.save_dir + "/log.txt", 'a') as f:
            f.write(f"Simulation stopped at {self.cur_timestamp}. \n")
            f.write(f"Total reward: " + str(self.total_reward) + "\n")

        # Save ablation results.
        np.save(self.save_dir + "/ablation_res.npy", self.ablation_result_img)
        im = Image.fromarray(self.ablation_result_img)
        im.save(self.save_dir + "/ablation_res.jpg")

        np.save(self.save_dir + "/temp_curve.npy", self.temp_lst)
        np.save(self.save_dir + "/time_curve.npy", self.time_lst)
        np.save(self.save_dir + "/cem_curve.npy", self.cem_lst)
        plt.plot(self.time_lst, self.cem_lst)
        plt.scatter(self.time_lst, self.cem_lst)
        plt.xlabel("Time (s)")
        plt.ylabel("Temperature (C)")
        plt.savefig('/home/yiwei/ConformalAblation/fig' + "/temp_curve2.png")

    def reset_button_clicked(self):
        self.status_bar.showMessage(f"Reset simulation.")

        self.angle_slider.setValue(0)
        state = self.env.reset(self.lesion_mask)

        self.loaded_traj = []

        self.temp_lst = []
        self.time_lst = []

        state_scaled_pixmap_list = np_to_pixmap(state, scale_factor=5)
        self.img_1.setPixmap(state_scaled_pixmap_list[0])
        self.img_2.setPixmap(state_scaled_pixmap_list[1])
        self.img_3.setPixmap(state_scaled_pixmap_list[2])

    def simulation_loop(self):
        while self.loop_flag:
            if self.replay_traj:
                if self.loaded_traj:
                    self.current_angle = int(self.loaded_traj.pop(0))
                else:
                    self.stop_button_clicked()
            else:
                self.time_label.setText(str(self.cur_timestamp))
                with open(self.save_dir + "/traj.txt", 'a') as f:
                    f.write(str(int(self.cur_timestamp)) + "," + \
                            str(self.current_angle) + "\n")

            state, reward, done, info = self.env.step(action=None, 
                                                        angle=self.current_angle,
                                                        power_switch=self.power_switch)
            temp_ctrl = state[1][50][51] / 4.5 + self.init_temp
            ablated = (state[2] == 0) | (state[2] == 255)
            s_ab = np.count_nonzero(ablated==1)

            self.temp_lst.append(temp_ctrl)
            self.cem_lst.append(s_ab)
            self.time_lst.append(self.cur_timestamp)
            if done:
                print("Termination condition met: " + str(info["termination_cause"]))
                self.stop_button_clicked()
                break
            self.ablation_result_img = state[-1]
            self.total_reward += reward
            self.cur_timestamp += self.time_step

            state_scaled_pixmap_list = np_to_pixmap(state, scale_factor=3)
            self.img_1.setPixmap(state_scaled_pixmap_list[0])
            self.img_2.setPixmap(state_scaled_pixmap_list[1])
            self.img_3.setPixmap(state_scaled_pixmap_list[2])

            time.sleep(0.5)

    def save_traj_button_clicked(self):
        self.status_bar.showMessage(f"Trajectory saved.")

    def angle_slicer_value_changed(self, value):
        self.angle_label.setText(str(value))
        self.current_angle = value

    def load_traj_button_clicked(self):
        traj_filename = self.traj_filename_edit.toPlainText()
        print("Loaded traj file: " + traj_filename)
        self.replay_traj = True
        with open(traj_filename) as f:
            lines = f.readlines()
            for line in lines:
                timestamp, rotation_angle = line[:-1].split(",")
                self.loaded_traj.append(rotation_angle)


# Utilities
def np_to_pixmap(np_imgs, scale_factor):
    scaled_pixmap_list = []
    for np_img in np_imgs:
        np_img = np_img / 255
        pil_img = Image.fromarray(np.uint8(cm.terrain(np_img) * 255))
        qt_img = ImageQt.ImageQt(pil_img)
        pixmap_img = QtGui.QPixmap.fromImage(qt_img)

        pixmap_size = pixmap_img.size()
        scaled_pixmap = pixmap_img.scaled(scale_factor * pixmap_size)
        scaled_pixmap_list.append(scaled_pixmap)

    return scaled_pixmap_list


if __name__ == "__main__":
    app = QApplication(sys.argv)
    UIWindow = ICAP_UI()
    app.exec_()
