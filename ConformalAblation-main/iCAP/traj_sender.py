import time

import numpy as np
from igtl_utlis import *

if __name__ == '__main__':
    robot_server_ip = "192.168.88.253"
    robot_server_port = 18936
    socket = launch_igtl_client(robot_server_ip, robot_server_port)

    traj_filename = r"traj_test.txt"
    traj = []
    with open(traj_filename) as f:
        lines = f.readlines()
        for line in lines:
            timestamp, rotation, insertion = line[:-1].split(",")
            traj.append([float(timestamp), float(rotation), float(insertion)])

    start_time = time.time()
    i = 0
    for i in range(len(lines)):
        while True:
            if time.time() - start_time >= traj[i][0]:
                print(time.time() - start_time)
                break
        send_test_string(socket, str(traj[i][1]) + "," + str(traj[i][2]))
        print("Sent:" + str(traj[i][0]) + "," + str(traj[i][1]))

    print(traj)
    pass
