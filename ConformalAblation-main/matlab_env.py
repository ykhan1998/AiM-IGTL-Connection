import glob
import random

import cv2
import imutils
import matlab.engine
import matplotlib.pyplot as plt
import numpy as np
import scipy.io
from gymnasium import Env, spaces
from gymnasium.spaces import Discrete

from constants import *
from lesion_pattern import LesionPattern
from utils import compute_status_map
from MRTI.src.mrti_utils import update_cem
import datetime
from matplotlib import cm
from PIL import Image


class MatlabEnv(Env):
    metadata = {"render_modes": ["human", "rgb_array"], "render_fps": 20}
    def __init__(self, env_config, verbose, plot, render_mode=None):
        super(MatlabEnv, self).__init__()
        
        # Directories
        self.project_path = WORK_DIR

        # Environment Configuration
        self.height = env_config["HEIGHT"]
        self.width = env_config["WIDTH"]
        self.init_temp = env_config["INIT_TEMP"]
        self.cem_temp = env_config["CEM_TEMP"]
        self.cem_thresh = env_config["CEM_THRESH"]
        self.temp_max = env_config["TEMP_MAX"]
        self.pa_max = env_config["PA_MAX"]
        self.pa_min = env_config["PA_MIN"]

        self.time_step = env_config["TIME_STEP"]
        self.angle_step = env_config["ANGLE_STEP"]

        # State initialization [pressure_field, temperature_map, status_map]
        self.state = np.zeros((3, self.height, self.width), dtype=np.uint8)

        self.current_angle = None
        self.last_angle = 0
        self.cur_step = 0
        self.max_step = env_config["MAX_STEP"]
        self.max_dmg = 100

        self.cem_map = np.zeros((self.height, self.width))

        self.scene_value_history = None

        self.observation_space = spaces.Box(low=0, 
                                            high=255, 
                                            shape=(3, self.height, self.width), 
                                            dtype=np.uint8)

        # Actions: Rotate -theta, 0, theta degree, Turn off.
        self.action_space = Discrete(3)
        # self.action_space = Discrete(4)
        
        self.lesion_mask = None
        self.lesion_size = 0

        # Initialize Matlab Engine
        self.eng = matlab.engine.start_matlab()
        self.eng.addpath(self.eng.genpath(MATLAB_FUNC_PATH), nargout=0)

        # Other options
        self.verbose = verbose
        self.plot = plot

        assert render_mode is None or render_mode in self.metadata["render_modes"]
        self.render_mode = render_mode

    def reset(self, lesion_mask, acpr_filename):
        if self.plot:
            fig, self.ax = plt.subplots(1, 3)
            plt.cla()
            self.ax[0].imshow(self.state[0])
            self.ax[1].imshow(self.state[1], vmin=0, vmax=255)
            self.ax[2].imshow(self.state[2], vmin=0, vmax=255)
        
        self.lesion_mask = lesion_mask
        self.lesion_size = np.count_nonzero(lesion_mask)

        # Load acoustic field
        self.acpr_filename = acpr_filename
        a_field = scipy.io.loadmat(self.acpr_filename)['acoustic_field']
        # Normalization the pressure amplitude 0 - 10^6 -> 0 - 255, then resize
        a_field_norm = ((a_field - self.pa_min) / (self.pa_max - self.pa_min) * 255)\
                        .astype('uint8')
        self.a_field = cv2.resize(a_field_norm, dsize=(self.height, self.width),
                                    interpolation=cv2.INTER_LINEAR)

        self.eng.resetTissue(self.acpr_filename, float(self.init_temp), \
                                float(self.cem_temp), nargout=0)

        temp_map = np.zeros((self.height, self.width), dtype=np.uint8)
        status_map = np.zeros((self.height, self.width), dtype=np.uint8)
        self.state = np.concatenate((self.a_field[np.newaxis, :, :],
                                     temp_map[np.newaxis, :, :],
                                     status_map[np.newaxis, :, :]), axis=0, dtype=np.uint8)

        self.cem_map = np.zeros((self.height, self.width))
        self.current_angle = 0
        self.last_angle = 0
        self.scene_value_history = np.zeros(2)
        self.cur_step = 0
        self.total_reward = 0
        
        return self.state
    
    def check_termination(self, status_map):
        info = {}
        done = False

        # Termination conditions
        if self.current_angle >= 360 or self.current_angle <= -360:
            done = True
            info["termination_cause"] = TERM_CAUSES["Completed_one_circle"]
        if self.cur_step >= self.max_step:
            done = True
            info["termination_cause"] = TERM_CAUSES["Reached_max step"]
        if  self.total_reward <= -self.max_dmg:
            done = True
            info["termination_cause"] = TERM_CAUSES["Too_much_damage"]
        if np.count_nonzero(status_map==STATUS_3) <= LESION_THRESH:
            done = True
            info["termination_cause"] = TERM_CAUSES["All_lesion_burnt"]
        
        if done:
            if self.plot:
                plt.close('all')
                # plt.pause(0.01)
            return True, info
        else:
            return False, info
        
    def step(self, action, angle, power_switch):
        # State [pressure_field; temperature_map; status_map]
        # Actions [0: rotate -1; 1: stay; 2: rotate 1]
        info = {}
        reward = 0
        done = False
        self.cur_step += 1

        if action is not None:
            rotate_angle = int(action - 1) * self.angle_step
            self.current_angle += rotate_angle
        elif angle is not None:
            rotate_angle = angle - self.last_angle
            self.last_angle = angle
            self.current_angle = angle
        else:
            raise Exception("No action or angle is given.")
        
        pressure_field = imutils.rotate(self.a_field, self.current_angle)

        temp_map = self.eng.oneStep(rotate_angle, power_switch, self.time_step, nargout=1)
        temp_map = np.array(temp_map).astype(float)
        temp_map = cv2.resize(temp_map, dsize=(self.height, self.width), 
                                interpolation=cv2.INTER_LINEAR)
        
        self.cem_map = update_cem(self.cem_temp, self.cem_map, temp_map, self.time_step/60)

        # cem_map = np.array(cem_map).astype(float)
        ablated_map = np.where(self.cem_map >= self.cem_thresh, 1, 0)
        ablated_map = cv2.resize(ablated_map, dsize=(self.height, self.width), 
                                interpolation=cv2.INTER_NEAREST)
        
        temp_map_norm = ((temp_map - self.init_temp) * 
                    (1 / (self.temp_max - self.init_temp) * 255))
        temp_map_norm[temp_map_norm > 255] = 255
        temp_map_norm = temp_map_norm.astype('uint8')

        status_map, status_value = compute_status_map(ablated_map, self.lesion_mask)
        status_value = round(status_value / self.lesion_size * 100, 3)

        done, info = self.check_termination(status_map)
        
        if done:
            return self.state, reward, done, info

        self.state = np.concatenate((pressure_field[np.newaxis, :, :],
                                     temp_map_norm[np.newaxis, :, :],
                                     status_map[np.newaxis, :, :]), axis=0, dtype=np.uint8)

        self.scene_value_history = np.append(self.scene_value_history, status_value)
        self.scene_value_history = self.scene_value_history[1:]

        # Compute reward for the current action
        reward = self.scene_value_history[-1] - self.scene_value_history[-2]
        self.total_reward += reward
        if self.verbose:
            print("Step: " + f"{self.cur_step:3d}" + \
                    ", Action: " + f"{action:1d}" + \
                    ", Current angle: " + f"{self.current_angle}" + \
                    ", Step Reward: " + f"{reward:2.2f}")

        if self.plot:
            self.ax[0].imshow(self.state[0])
            self.ax[1].imshow(self.state[1], vmin=0, vmax=255)
            self.ax[2].imshow(self.state[2], vmin=0, vmax=255, cmap='terrain')
            plt.pause(0.01)

        return self.state, reward, done, info

    def render(self, mode='rgb_array'):
        if mode == 'rgb_array':
            render_size = 600
            frame1 = cv2.resize(self.state[0], dsize=(render_size, render_size), 
                                interpolation=cv2.INTER_NEAREST)
            frame2 = cv2.resize(self.state[1], dsize=(render_size, render_size), 
                                interpolation=cv2.INTER_NEAREST)
            frame3 = cv2.resize(self.state[2], dsize=(render_size, render_size), 
                                interpolation=cv2.INTER_NEAREST)  
            frame1 = frame1 / 255
            frame2 = frame2 / 255
            frame3 = frame3 / 255
            frame1 = np.uint8(cm.terrain(frame1) * 255)[:, : , :3]
            frame2 = np.uint8(cm.terrain(frame2) * 255)[:, : , :3]
            frame3 = np.uint8(cm.terrain(frame3) * 255)[:, : , :3]
            colored_frame = np.concatenate((frame1, frame2, frame3), axis=1)
            return colored_frame # return RGB frame suitable for video
        elif mode == 'human':
            print("render.human")
        else:
            super(MatlabEnv, self).render(mode=mode)

    
