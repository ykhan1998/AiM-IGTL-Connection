import time

import numpy as np
import pyigtl

# import openigtlink as igtl


def launch_openigtlink_server(port):
    socket = igtl.ServerSocket.New()
    socket.SetReceiveTimeout(1)  # Milliseconds
    r = socket.CreateServer(port)
    if r < 0:
        print("Failed to create a server socket")

    while True:
        server_socket = socket.WaitForConnection(1000)
        if server_socket.IsNotNull():
            print("Found a IGTL client and connected.")
            break
        else:
            print("Cannot connect.")
            time.sleep(1)

    return server_socket


def launch_openigtlink_client(server_ip, server_port):
    client_socket = igtl.ClientSocket.New()
    client_socket.SetReceiveTimeout(1)  # Milliseconds
    ret = client_socket.ConnectToServer(server_ip, int(server_port))
    if ret == 0:
        print('Connection successful.')
    else:
        print('Could not connect to the server.')

    return client_socket


def send_test_string(socket, test_string):
    stringMsg = igtl.StringMessage.New()
    stringMsg.SetDeviceName("TestString")

    stringMsg.SetString(test_string)
    stringMsg.Pack()
    socket.Send(stringMsg.GetPackPointer(), stringMsg.GetPackSize())


def launch_pyigtl_server(port): 
    pyigtl_server = pyigtl.OpenIGTLinkServer(port, local_server=True)

    while True:
        if pyigtl_server.is_connected():
            print(f"Server connected. Port: {port}")
            break
        else:
            print(f"Waiting client connection. Port: {port}")
            time.sleep(1)

    return pyigtl_server


def launch_pyigtl_client(server_ip, port):
    pyigtl_client = pyigtl.OpenIGTLinkClient(server_ip, port)

    while True:
        if pyigtl_client.is_connected():
            print(f"Connected to server {server_ip}:{port}")
            break
        else:
            print(f"Waiting server connection {server_ip}:{port}")
            time.sleep(1)

    return pyigtl_client


def send_str_to_server(pyigtl_client, str_to_send, device_name, timestamp=None):
    string_message = pyigtl.StringMessage(str_to_send, device_name=device_name, timestamp=timestamp)
    pyigtl_client.send_message(string_message, wait=True)


def send_str_to_client(pyigtl_server, str_to_send, device_name, timestamp=None):
    string_message = pyigtl.StringMessage(str_to_send, device_name=device_name, timestamp=timestamp)
    pyigtl_server.send_message(string_message, wait=True)


def receive_str_from_client(pyigtl_server, device_name):
    string_message = pyigtl_server.receive_message(device_name=device_name)
    return string_message

def receive_str_from_server(pyigtl_client, device_name, timeout=3):
    message = pyigtl_client.wait_for_message(device_name, timeout)
    return message

def pyigtl_img_streaming_test_server(port):
    pyigtl_server = launch_pyigtl_server(port)

    image_size = [500, 300]
    radius = 60
    timestep = 0
    while True:
        timestep += 1
        voxels, cx, cy = generate_test_image(image_size, radius, timestep)
        print(f"time: {timestep}   position: ({cx}, {cy})")

        image_message = pyigtl.ImageMessage(voxels, device_name="TestImageFromServer")
        pyigtl_server.send_message(image_message, wait=True)

        time.sleep(3)


def pyigtl_img_streaming_test_client(server_ip, port):
    pyigtl_client = pyigtl.OpenIGTLinkClient(server_ip, port)

    image_size = [500, 300]
    radius = 60

    timestep = 0
    while True:
        timestep += 1
        voxels, cx, cy = generate_test_image(image_size, radius, timestep)
        print(f"time: {timestep}   position: ({cx}, {cy})")

        image_message = pyigtl.ImageMessage(voxels, device_name="TestImageFromClient")
        pyigtl_client.send_message(image_message, wait=True)


def generate_test_image(image_size, radius, timestep):
    cx = radius + (image_size[0] - 2 * radius) * 0.5 * (1 + np.sin(timestep * 0.05))
    cy = radius + (image_size[1] - 2 * radius) * 0.5 * (1 + np.sin(timestep * 0.06))
    
    y, x = np.ogrid[:image_size[0], :image_size[1]]
    mask = ((x - cx)**2 + (y - cy)**2) <= radius**2
    
    voxels = np.ones((image_size[0], image_size[1], 1), dtype=np.float32)
    voxels[mask] = 55.5
    
    # Add some noise to make the image more realistic
    noise = np.random.normal(0, 0.1, voxels.shape)
    voxels += noise
    voxels = np.clip(voxels, 0, 255)  # Ensure values are within valid range

    return voxels, cx, cy


if __name__ == "__main__":
    # Test Configuration
    server_ip = "localhost"
    port = 18944
    device_name = "TestDevice"

    test_mode = "client" # "server" or "client" 

    # Connection
    if test_mode == "server":
        pyigtl_server = launch_pyigtl_server(port)
    elif test_mode == "client":
        pyigtl_client = launch_pyigtl_client(server_ip, port)

    # Test sending string message
    if test_mode == "server":
        str_to_send_to_client = "Hello from server"
        send_str_to_client(pyigtl_server, str_to_send_to_client, device_name)
    elif test_mode == "client":
        str_to_send_to_server = "Hello from client"
        send_str_to_server(pyigtl_client, str_to_send_to_server, device_name)

    # # Server Test image streaming using pyigtl
    # if test_mode == "server":   
    #     pyigtl_img_streaming_test_server(port)
    # elif test_mode == "client":
    #     pyigtl_img_streaming_test_client(server_ip, port)

