import pyigtl
import datetime
from scipy.spatial.transform import Rotation as R
import numpy as np
import time

# Send mesh to OpenIGTLink server
class commandclient:
    def __init__(self, ip='192.168.88.250', port=18936):
        self.ip = ip
        self.port = port

    def rot_geodesic_angle(self, R1, R2):
        # Relative rotation R_rel = R1^T R2
        R_rel = R1.T @ R2
        # Numerical safety: clip trace-based cosine into [-1, 1]
        cos_theta = (np.trace(R_rel) - 1.0) * 0.5
        cos_theta = np.clip(cos_theta, -1.0, 1.0)
        return np.arccos(cos_theta)  # radians
    
    def transforms_close(self, T1, T2, tol_trans=1e-1, tol_rot=1e-1):
        """
        T1, T2: 4x4 homogeneous transforms
        tol_trans: meters (or your length unit)
        tol_rot: radians
        """
        # Extract R, t
        R1, t1 = T1[:3, :3], T1[:3, 3]
        R2, t2 = T2[:3, :3], T2[:3, 3]
        # Translation difference
        trans_err = np.linalg.norm(t1 - t2)
        # Rotation geodesic error
        rot_err = self.rot_geodesic_angle(R1, R2)
        print(trans_err,rot_err)
        return ((trans_err <= tol_trans) and (rot_err <= tol_rot))
    
    def connect(self):
        self.client = pyigtl.OpenIGTLinkClient(self.ip, self.port)
        print('connected')

    def generateTimestamp(self):
        return (int(datetime.datetime.now().timestamp()))

    def start_up(self):
        msg = "START_UP"
        timeStamp = self.generateTimestamp()
        output_message = pyigtl.StringMessage(string=msg, timestamp=timeStamp, device_name='start_up')
        self.client.send_message(output_message)
        print("Sent Message:Start_up")
        message = self.client.wait_for_message("measured_cp")
        self.current_cp = message.matrix
        print("Received Message:measured_cp")
        print(self.current_cp)
        # Or we can get all the latest messages
        # message = self.client.get_latest_messages()
        # print(message)
        # for m in message:
        #     if m.device_name == "measured_cp":
        #         self.current_cp = m.matrix
        #         print(m)


    def image_listener(self, n_slice):
        self.image_list = []
        i = 0
        while i<n_slice:
            img_msg = self.client.wait_for_message("Image")
            self.image_list.append(img_msg)
            i += 1

    def trajectory_generation(self, area, tip_location, n_step):
        #TODO: Generate a trajectory based on area input
        #Input: desired abation image/area, tip_location on the image, number of steps wanted
        #Output: A 4x4xn array, n items list of 4x4 traget transformation matrix
        #        A 1xn   array/list, n items time map defines how long does it has to stay at a specific location
        # return trajectory, time map
        return 0

    def generate_circle(self, n_steps=90):
        #generate a 5s per step clock wise circle trajectory
        trajectory = []
        angles = np.linspace(0, 360, n_steps, endpoint=False)  # yaw angles in degrees
        rot = R.from_matrix(self.current_cp[0:3, 0:3])
        for yaw in angles:
            # Relative yaw rotation around Z-axis
            R_yaw = R.from_euler('z', yaw, degrees=False)
            # Compose with starting orientation
            R_new = rot * R_yaw
            T_new = self.current_cp.copy()
            T_new[0:3, 0:3] = R_new.as_matrix().tolist()
            trajectory.append(T_new)
        timeMap = np.full(90, 2)
        return trajectory, timeMap
    
    def rotate_tip(self, target):
        timeStamp = self.generateTimestamp()
        while not self.transforms_close(self.current_cp, target):
            output_message = pyigtl.TransformMessage(matrix=target, timestamp=timeStamp, device_name='TARGET_POINT')
            self.client.send_message(output_message)
            message = self.client.wait_for_message("measured_cp")
            self.current_cp = message.matrix

    
    def insert_tip(self, insert_depth):
        target = self.current_cp
        target[2,3] = target[2,3] + insert_depth
        timeStamp = self.generateTimestamp()
        output_message = pyigtl.TransformMessage(matrix=target, timestamp=timeStamp, device_name='TARGET_POINT')
        self.client.send_message(output_message)
        while not self.transforms_close(self.current_cp, target):
            message = self.client.wait_for_message("measured_cp")
            self.current_cp = message.matrix

    
    def run(self, n_step, dist_step):
        # The supposing workflow
        #First get all images
        self.connect()
        self.start_up()
        self.image_listener(n_step)
        #Afer getting all the images, calculate the trajectory need for each one
        i = 0
        trajectory = []
        timeMap = []
        # Start the ablation slicer by slice
        j = 0
        while i < n_step:
            trajectory,timeMap = self.trajectory_generation(self.image_list(i),self.current_cp,n_step)
            input("Turn on the ablator and Press Enter to continue...")
            for step, duration in zip(trajectory,timeMap):
                self.rotate_tip(step)
                time.sleep(duration)
            self.insert_tip(dist_step)
            trajectory = []
            timeMap = []
            i += 1

    def test_run(self):
        self.connect()
        self.start_up()
        trajectory,timeMap = self.generate_circle()
        for step, duration in zip(trajectory,timeMap):
            print(step,duration)
            self.rotate_tip(step)
            time.sleep(duration)
        self.insert_tip(0.5)

if __name__ == "__main__":
    v = commandclient()
    v.test_run()