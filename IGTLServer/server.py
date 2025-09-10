import openigtlink as igtl
import datetime
import time

class IGTLServer():

    def __init__(self, ip='192.168.88.250', port=18936):
        self.ip = ip,
        self.port = port
        self.initialize()

    def initialize(self):
        print("Initializing IGTL Server...")
        self.transMsg = igtl.TransformMessage.New()
        self.headerMsg = igtl.MessageBase.New()
        self.stringMsg = igtl.StringMessage.New()
        # The following variables are used to regulate the incoming messages.
        ret = self.connect(self.ip, self.port)
        if ret == 1:
            print( "Connection successful.")
        else:
            print("Could not connect to the server.")

    def generateTimestampNameID(self, prefix):
        timestampID = [prefix, "_"]
        currentTime = datetime.datetime.now()
        timestampID.append(currentTime.strftime("%H%M%S%f"))
        timestampIDname = ''.join(timestampID)
        return timestampIDname

    def process(self):
        # Initialize receive buffer
        self.headerMsg = igtl.MessageBase.New()    
        self.headerMsg.InitPack()
        timeout = False
        [result, timeout] = self.clientServer.Receive(self.headerMsg.GetPackPointer(), self.headerMsg.GetPackSize(), timeout)
        if (result != self.headerMsg.GetPackSize()):
            print("Incorrect pack size!")
            return
        # Deserialize the header
        self.headerMsg.Unpack()
        # Check data type and respond accordingly
        msgType = self.headerMsg.GetDeviceType()
        if msgType != '':
            print("Recieved: %s" % msgType)
        # ---------------------- TRANSFORM ----------------------------
        if (msgType == "TRANSFORM"):
            self.transMsg = igtl.TransformMessage.New()
            self.transMsg.Copy(self.headerMsg.GetPointer()) # Can't get MessageHeaders to instantiate, but SetMessageHeader seems to just be calling Copy
            self.transMsg.AllocatePack()
            # Receive transform data from the socket

            [r, timeout] = self.clientServer.Receive(self.transMsg.GetPackBodyPointer(), self.transMsg.GetPackBodySize(), timeout)
            self.transMsg.Unpack()
            self.onReceiveTransform(self.transMsg)
        # ---------------------- STRING ----------------------------
        elif (msgType == "STRING"):
            #Create a message buffer to receive string data
            self.stringMsg = igtl.StringMessage.New()
            self.stringMsg.Copy(self.headerMsg.GetPointer()) # Can't get MessageHeaders to instantiate, but SetMessageHeader seems to just be calling Copy
            self.stringMsg.AllocatePack()
            # Receive string data from the socket
            [r, timeout] = self.clientServer.Receive(self.stringMsg.GetPackBodyPointer(), self.stringMsg.GetPackBodySize(), timeout)
            self.stringMsg.Unpack()
            self.onReceiveString(self.stringMsg)

    def connect(self, ip, port):
        self.clientServer = igtl.ServerSocket.New()
        self.clientServer.SetReceiveTimeout(1) # Milliseconds
        ret = self.clientServer.CreateServer(ip,int(port))
        if ret == 0:
            return True
        else:
            return False

    def sendTransform(self, data, name):
        #Input should be array
        timestampIDname = self.generateTimestampNameID("TGT")
        transformSendMsg = igtl.TransformMessage.New()
        # The following two might have error and not needed
        transformSendMsg.AllocatePack()
        transformSendMsg.SetTimeStamp(timestampIDname)
        transformSendMsg.SetDeviceName(name)
        matrix4x4 = data.tolist()
        transformSendMsg.SetMatrix(matrix4x4)
        transformSendMsg.Pack()
        self.clientServer.Send(transformSendMsg.GetPackPointer(), transformSendMsg.GetPackSize())
    
    def onReceiveTransform(self,transMsg):
        matrix4x4 = [[1.0, 0.0, 0.0, 0.0], [0.0, 1.0, 0.0, 0.0], [0.0, 0.0, 1.0, 0.0], [0.0, 0.0, 0.0, 1.0]]
        matrix4x4 = transMsg.GetMatrix(matrix4x4)
        name = transMsg.GetDeviceName()
        print(name, matrix4x4)
        return name, matrix4x4

    def sendString(self,data,name):
        timestampIDname = self.generateTimestampNameID("CMD")
        stringSendMsg = igtl.StringMessage.New()
        stringSendMsg.AllocatePack()
        stringSendMsg.SetTimeStamp(timestampIDname)
        stringSendMsg.SetDeviceName(name)
        stringSendMsg.SetString(data)
        stringSendMsg.SetEncoding(3)
        stringSendMsg.Pack()
        self.clientServer.Send(stringSendMsg.GetPackPointer(), stringSendMsg.GetPackSize())

    def onReceiveString(self, headerMsg):
        name = headerMsg.GetDeviceName()
        string = headerMsg.GetString()
        print(name, string)
        return name, string

    def sendPoint(self,data,name):
        timestampIDname = self.generateTimestampNameID("TGT")
        pointMsg = igtl.PointMessage.New()
        pointMsg.AllocatePack()
        pointMsg.SetTimeStamp(timestampIDname)
        pointMsg.SetDeviceName(name)
        point0 = igtl.PointElement.New()
        # Try to minimize the 
        #point0.SetName(name)
        #point0.SetGroupName(name)
        #point0.SetRGBA(0xFF, 0x00, 0x00, 0xFF)
        # Or maybe send via list?
        point0.SetPosition(data[0], data[1], data[2])
        #point0->SetRadius(15.0);
        #point0->SetOwner("IMAGE_0");

    def disconnectOpenIGTEvent(self):
        self.clientServer.CloseSocket()


if __name__ == "__main__":
    testClass = IGTLServer()

