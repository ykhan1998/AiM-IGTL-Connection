import openigtlink as igtl
import datetime
import time

class IGTLListener():

    def __init__(self, ip='127.0.0.1', port=18936, *args):
            self.parameter = {
                'ip': ip,
                'port': port
            }


    def initialize(self):
        print("Initializing IGTL Listener...")
        self.transMsg = igtl.TransformMessage.New()
        self.headerMsg = igtl.MessageBase.New()
        self.stringMsg = igtl.StringMessage.New()
        # The following variables are used to regulate the incoming messages.
        self.prevImgTime = 0.0
        self.prevTransMsgTime = 0.0
        self.minTransMsgInterval = 0.1 # 10 Hz
        self.pendingTransMsg = False

        socketIP = self.parameter['ip']
        socketPort = self.parameter['port']
        ret = self.connect(socketIP, socketPort)
        if ret == 0:
            print( "Connection successful.")
        else:
            print("Could not connect to the server.")

    def generateTimestampNameID(self, last_prefix_sent):
        timestampID = [last_prefix_sent, "_"]
        currentTime = datetime.datetime.now()
        timestampID.append(currentTime.strftime("%H%M%S%f"))
        timestampIDname = ''.join(timestampID)
        return timestampIDname

    def process(self):

        self.minTransMsgInterval = 0.1 # 100ms

        # Initialize receive buffer
        self.headerMsg = igtl.MessageBase.New()    
        self.headerMsg.InitPack()

        self.clientServer.SetReceiveTimeout(10) # Milliseconds
        #self.clientServer.SetReceiveTimeout(int(self.minTransMsgInterval*1000.0)) # Milliseconds
        timeout = True
        [result, timeout] = self.clientServer.Receive(self.headerMsg.GetPackPointer(), self.headerMsg.GetPackSize(), timeout)

        ## TODO: timeout always return True - is it a bug?

        ## TODO: Need to detect disconnection
        #if result == 0:
        #  self.clientServer.CloseSocket()
        #  #self.closeSocketSignal.emit()
        #  return
        msgTime = time.time()

        if result==0 and timeout:
            # Time out
            if self.pendingTransMsg:
                if msgTime - self.prevTransMsgTime > self.minTransMsgInterval:
                    print("Sending out pending transform.")
                    self.transMsg.Unpack()
                    self.onReceiveTransform(self.transMsg)
                    self.prevTransMsgTime = msgTime
                    self.pendingTransMsg = False
            return
            
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
            timeout = False
            [r, timeout] = self.clientServer.Receive(self.transMsg.GetPackBodyPointer(), self.transMsg.GetPackBodySize(), timeout)

            # Check the time interval. Send the transform to MRI only if there was enough interval.
            if msgTime - self.prevTransMsgTime > self.minTransMsgInterval:
                self.transMsg.Unpack()
                self.onReceiveTransform(self.transMsg)
                self.prevTransMsgTime = msgTime
                self.pendingTransMsg = False
            else:
                self.pendingTransMsg = True
            
        # ---------------------- STRING ----------------------------
        elif (msgType == "STRING"):
            #Create a message buffer to receive string data
            self.stringMsg = igtl.StringMessage.New()


            self.stringMsg.Copy(self.headerMsg.GetPointer()) # Can't get MessageHeaders to instantiate, but SetMessageHeader seems to just be calling Copy
            self.stringMsg.AllocatePack()

            # Receive string data from the socket
            timeout = False
            [r, timeout] = self.clientServer.Receive(self.stringMsg.GetPackBodyPointer(), self.stringMsg.GetPackBodySize(), timeout)
            self.stringMsg.Unpack()
            self.onReceiveString(self.stringMsg)

        elif (msgType == "POINT"):
            self.pointMsg = igtl.PointMessage.New()
            self.pointMsg.Copy(self.headerMsg.GetPointer())
            self.pointMsg.AllocatePack()
            timeout = False
            [r, timeout] = self.clientServer.Receive(self.pointMsg.GetPackBodyPointer(), self.pointMsg.GetPackBodySize(), timeout)
            self.pointMsg.Unpack()

        endTime = time.time()
        sleepTime = self.minTransMsgInterval - (endTime - msgTime)
        print('sleep time = ' + str(sleepTime))
        if sleepTime < 0:
            sleepTime = 0


    def connect(self, ip, port, type='server'):
        if type == 'client':
            self.clientServer = igtl.ClientSocket.New()
            self.clientServer.SetReceiveTimeout(1) # Milliseconds
            #self.clientServer.SetReceiveBlocking(0)
            #self.clientServer.SetSendBlocking(0)
            ret = self.clientServer.ConnectToServer(ip,int(port))
        elif type == 'server':
            self.clientServer = igtl.ServerSocket.New()
            self.clientServer.SetReceiveTimeout(1) # Milliseconds
            ret = self.clientServer.CreateServer(ip,int(port))

        if ret == 0:
            print("Connection successful")
            return True
        else:
            print("Connection failed")
            return False

    def packTransform(self, data):
        igtl::TransformMessage::Pointer transformSendMsg = igtl::TransformMessage::New();
        transformSendMsg->SetHeaderVersion(IGTL_HEADER_VERSION_1);
        transformSendMsg->AllocatePack();
        transformSendMsg->SetTimeStamp(0, 1234567892);
        transformSendMsg->SetDeviceName("DeviceName");
        transformSendMsg->SetMatrix(inMatrix);
        transformSendMsg->Pack();
    
    def onReceiveTransform(self,transMsg):
        matrix4x4 = [[1.0, 0.0, 0.0, 0.0], [0.0, 1.0, 0.0, 0.0], [0.0, 0.0, 1.0, 0.0], [0.0, 0.0, 0.0, 1.0]]
        matrix4x4 = transMsg.GetMatrix(matrix4x4)
        name = transMsg.GetDeviceName()
        print(name, matrix4x4)
        return name, matrix4x4

    def packString(self,data):
        igtl::StringMessage::Pointer stringSendMsg = igtl::StringMessage::New();
        stringSendMsg->SetHeaderVersion(IGTL_HEADER_VERSION_1);
        stringSendMsg->SetString(IGTL_STRING_TEST_STRING);
        stringSendMsg->AllocatePack();
        stringSendMsg->SetTimeStamp(0, 1234567892);
        stringSendMsg->SetDeviceName("DeviceName");
        stringSendMsg->SetEncoding(3);
        stringSendMsg->Pack();

    def onReceiveString(self, headerMsg):
        name = headerMsg.GetDeviceName()
        string = headerMsg.GetString()
        print(name, string)
        return name, string

    def packPoint(self,data):
        igtl::PointMessage::Pointer pointMsg;
        pointMsg = igtl::PointMessage::New();
        pointMsg->SetDeviceName("PointSender");
        
        //---------------------------
        // Create 1st point
        igtl::PointElement::Pointer point0;
        point0 = igtl::PointElement::New();
        point0->SetName("POINT_0");
        point0->SetGroupName("GROUP_0");
        point0->SetRGBA(0xFF, 0x00, 0x00, 0xFF);
        point0->SetPosition(10.0, 20.0, 30.0);
        point0->SetRadius(15.0);
        point0->SetOwner("IMAGE_0");

    def onReceivePoint(self, pointMsg):
        name = pointMsg.GetDeviceName()
        pointelement = igtl.PointElement.New()
        pointMsg.GetPointElement(0,pointelement)
        pointarray = pointelement.GetPosition(pointarray)
        print(name, pointarray)
        return name, pointarray

        
    def disconnectOpenIGTEvent(self):
        self.clientServer.CloseSocket()

    def sendCommand(self, command):
        timestampIDname = self.generateTimestampNameID("CMD")
        startupmsg = igtl.StringMessage.New()
        startupmsg.SetDeviceName(timestampIDname)
        startupmsg.SetString(command)
        startupmsg.SetEncoding(3)
        startupmsg.Pack()
        self.clientServer.Send(startupmsg.GetPackPointer(), startupmsg.GetPackSize())

    def sendTarget(self, target):
        timestampIDname = self.generateTimestampNameID("TGT")
        targetmsg = igtl.TransformMessage.New()
        targetmsg.SetDeviceName(timestampIDname)
        matrix4x4 = target.tolist()
        targetmsg.SetMatrix(matrix4x4)
        targetmsg.Pack()
        self.clientServer.Send(targetmsg.GetPackPointer(), targetmsg.GetPackSize())
        

    def StartUp(self):
        self.sendCommand("START_UP")
        i = 1
        while i!=12:
            self.process()
            i+=1

if __name__ == "__main__":
    testClass = IGTLListener()
    testClass.initialize()
    testClass.StartUp()
