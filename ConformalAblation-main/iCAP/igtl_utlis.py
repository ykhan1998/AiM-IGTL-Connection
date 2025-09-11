import time

from grpc import server

# import openigtlink as igtl
import pyigtl


def launch_igtl_server(port, py=True, local_server=True):
    if py:
        server_socket = pyigtl.OpenIGTLinkServer(port, local_server)
        while True:
            if not server_socket.is_connected():
                print("No client is connected.")
                time.sleep(1)
            else:
                print("Connected to a client.")
                break
    else:
        socket = igtl.ServerSocket.New()
        socket.SetReceiveTimeout(1)  # Milliseconds
        r = socket.CreateServer(port)
        if r < 0:
            print("Failed to create a server socket")

        while True:
            server_socket = socket.WaitForConnection(1000)
            if server_socket.IsNotNull():
                print("Connected to a client.")
                break
            else:
                print("No client is connected.")
                time.sleep(1)

    return server_socket


def launch_igtl_client(server_ip, server_port, py=True):
    if py:
        igtl_client = pyigtl.OpenIGTLinkClient(server_ip, server_port)
        while True:
            if not igtl_client.is_connected():
                time.sleep(1)
                print("No server is connected.")
            else:
                print("Connected to a server.")
                break
    else:
        igtl_client = igtl.ClientSocket.New()
        igtl_client.SetReceiveTimeout(1)  # Milliseconds
        ret = igtl_client.ConnectToServer(server_ip, int(server_port))
        if ret == 0:
            print('Connected to a server.')
        else:
            print('No server is connected.')

    return igtl_client


def send_test_string(socket, test_string, py=True):
    if py:
        string_message = pyigtl.StringMessage(test_string, device_name="Text")
        socket.send_message(string_message)
    else:
        stringMsg = igtl.StringMessage.New()
        stringMsg.SetDeviceName("TestString")

        stringMsg.SetString(test_string)
        stringMsg.Pack()
        socket.Send(stringMsg.GetPackPointer(), stringMsg.GetPackSize())




if __name__ == "__main__":
    ip = "localhost"
    port = 18944
    socket = launch_igtl_server(port)
    # socket = launch_igtl_client(ip, port)

    test_string = "Hello world!"
    send_test_string(socket, test_string)

    pass