import argparse, sys, shutil, os, logging
import numpy as np
import sqlite3
import pydicom
import openigtlink as igtl
import time


#  Usage:
#
#  $ python splitSeriesByTag.py  [-h] [-r] TAG [TAG ...] SRC_DIR DST_DIR
#
#  Aarguments:
#         TAG:       DICOM Tag (see below)
#         SRC_DIR:   Source directory that contains DICOM files.
#         DST_DIR:   Destination directory to save NRRD files.
#
#  Dependencies:
#  This script requires 'pydicom' and 'pynrrd.'
#
#  Examples of DICOM Tags:
#   - General
#    - (0008,103e) : SeriesDescription
#    - (0010,0010) : PatientsName
#    - (0020,0010) : StudyID
#    - (0020,0011) : SeriesNumber
#    - (0020,0037) : ImageOrientationPatient
#    - (0008,0032) : AcquisitionTime
#
#   - Siemens MR Header
#    - (0051,100f) : Coil element (Siemens)
#    - (0051,1016) : Real/Imaginal (e.x. "R/DIS2D": real; "P/DIS2D": phase)


#
# Match DICOM attriburtes
#
def getDICOMAttribute(path, tags):
    dataset = None
    try:
        dataset = pydicom.dcmread(path, specific_tags=None)
    except pydicom.errors.InvalidDicomError:
        print("Error: Invalid DICOM file: " + path)
        return None

    insertStr = ''
    for tag in tags:
        key = tag.replace(',', '')
        if key in dataset:
            element = dataset[key]
            if insertStr == '':
                insertStr = "'" + str(element.value) + "'"
            else:
                insertStr = insertStr + ',' + "'" + str(element.value) + "'"

    return insertStr;


#
# Convert attribute to folder name (Remove special characters that cannot be
# included in a path name)
#
def removeSpecialCharacter(v):
    input = str(v)  # Make sure that the input parameter is a 'str' type.
    removed = input.translate({ord(c): "-" for c in "!@#$%^&*()[]{};:/<>?\|`="})

    return removed


#
# Concatenate column names
#
def concatColNames(tags):
    r = ''
    for tag in tags:
        key = tag.replace(',', '')
        if r == '':
            r = 'x' + key + ' text'
        else:
            r = r + ',' + 'x' + key + ' text'
    return r;


#
# Build a file path database based on the DICOM tags
#
def buildFilePathDBByTags(con, srcDir, tags, fRecursive=True):
    # Create a table
    con.execute('CREATE TABLE dicom (' + concatColNames(tags) + ',path text)')

    filePathList = []
    postfix = 0
    attrList = []

    print("Processing directory: %s..." % srcDir)

    for root, dirs, files in os.walk(srcDir):
        for file in files:
            srcFilePath = os.path.join(root, file)
            insertStr = getDICOMAttribute(srcFilePath, tags)
            if insertStr == None:
                print("Could not obtain attributes for %s" % srcFilePath)
                continue
            else:
                # Add the file path
                insertStr = insertStr + ',' + "'" + srcFilePath + "'"
                con.execute('INSERT INTO dicom VALUES (' + insertStr + ')')
                # print('INSERT INTO dicom VALUES (' + insertStr + ')')

        if fRecursive == False:
            break

    con.commit()


def export_imgmsg(image, dataset, device_name, sliceSpacing):
    try:
        rescaleIntercept = dataset['00281052'].value  # RescaleIntercept (b)
        rescaleSlope = dataset['00281053'].value  # RescaleSlope     (m)  U = m*SV + b
    except KeyError:
        rescaleIntercept = 0
        rescaleSlope = 1

    slice = {
            'position': np.array(dataset['00200032'].value),  # ImagePositionPatient
            'orientation': np.array(dataset['00200037'].value),  # ImageOrientationPatient
            'spacing': np.array(dataset['00280030'].value),  # PixelSpacing
            'sliceThickness': dataset['00180050'].value,  # SliceThickness
            'rows': dataset['00280010'].value,  # Rows
            'columns': dataset['00280011'].value,  # Columns
            'sliceLocation': dataset['00201041'].value,  # SliceLocation
            'bitsAllocated': dataset['00280100'].value,  # BitsAllocated
            'instanceNumber': dataset['00200013'].value,  # InstanceNumber -- image number
            'img_type': dataset['0043102F'].value  # 0 for magnitude img, 1 for phase img
        }

    if rescaleIntercept >= 0:
        slice['pixelArray'] = dataset.pixel_array * rescaleSlope + rescaleIntercept
    else:
        slice['pixelArray'] = (dataset.pixel_array * rescaleSlope + rescaleIntercept).astype('int16')

    dim = np.array([slice['columns'], slice['rows'], 1])

    spacing = slice['spacing']
    spacing = np.append(spacing, sliceSpacing)

    # Image orientation matrix
    # The orientation matrix is defined as N = [n1, n2, n3], where n1, n2, and n3 are
    # normal column vectors representing the directions of the i, j, and k indicies.
    norm = slice['orientation'].reshape((2, 3))
    norm = np.append(norm, np.cross(norm[0], norm[1]).reshape((1, 3)), axis=0)
    norm = np.transpose(norm)

    # Switch from LPS to RAS
    lpsToRas = np.transpose(np.array([[-1., -1., 1.]]))
    norm = norm * lpsToRas

    # Location of the first voxel in RAS
    pos = slice['position'].reshape(3, 1) * lpsToRas

    # Location of the the image center
    # OpenIGTLink uses the location of the volume center while SliceLocation in DICOM is the position of the first voxel.
    offset = norm * (spacing * (dim - 1.0) / 2.0)
    pos = pos + offset[:, [0]] + offset[:, [1]] + offset[:, [2]]

    nbytes = slice['bitsAllocated'] / 8
    # image = image.astype(np.int8)
    # image = image[:, np.newaxis]
    # image = np.transpose(image, axes=(2, 1, 0))
    typestr = str(image.dtype)

    # seriesDescription       = dataset['0008103e'].value # (0008,103e) : SeriesDescription
    # patientsName            = dataset['00100010'].value # (0010,0010) : PatientsName
    # studyID                 = dataset['00200010'].value # (0020,0010) : StudyID
    seriesNumber = dataset['00200011'].value  # (0020,0011) : SeriesNumber
    # imageOrientationPatient = dataset['00200037'].value # (0020,0037) : ImageOrientationPatient
    # acquisitionTime         = dataset['00080032'].value # (0008,0032) : AcquisitionTime

    ## Create an OpenIGTLink message
    DataTypeTable = {
        'int8': [2, 1],  # TYPE_INT8    = 2, 1 byte
        'uint8': [3, 1],  # TYPE_UINT8   = 3, 1 byte
        'int16': [4, 2],  # TYPE_INT16   = 4, 2 bytes
        'uint16': [5, 2],  # TYPE_UINT16  = 5, 2 bytes
        'int32': [6, 4],  # TYPE_INT32   = 6, 4 bytes
        'uint32': [7, 4],  # TYPE_UINT32  = 7, 4 bytes
        'float32': [10, 4],  # TYPE_FLOAT32 = 10, 4 bytes
        'float64': [11, 8],  # TYPE_FLOAT64 = 11, 8 bytes
    }

    imageMsg = igtl.ImageMessage.New()
    # imageMsg.SetDimensions(data.shape[0], data.shape[1], data.shape[2])
    imageMsg.SetDimensions(int(dim[0]), int(dim[1]), int(dim[2]))

    typeid = 0
    if typestr in DataTypeTable:
        typeid = DataTypeTable[typestr][0]
    else:
        print('Data type: %s is not compatible with OpenIGTLink.' % str(data.dtype))
        return

    imageMsg.SetScalarType(typeid)

    imageMsg.SetDeviceName(device_name)
    imageMsg.SetNumComponents(1)
    imageMsg.SetEndian(2)  # little is 2, big is 1
    imageMsg.SetSpacing(spacing[0], spacing[1], spacing[2])
    imageMsg.AllocateScalars()

    matrixNP = np.concatenate((norm, pos), axis=1)
    matrixNP = np.concatenate((matrixNP, np.array([[0, 0, 0, 1]])), axis=0)
    matrix = matrixNP.tolist()

    imageMsg.SetMatrix(matrix)
    img = bytes(image)

    igtl.copyBytesToPointer(img, igtl.offsetPointer(imageMsg.GetScalarPointer(), 0))

    imageMsg.Pack()

    return imageMsg
    

    

def exportToIGTL(filelist, sock=None, imgname=None):
    nSlices = len(filelist)

    # Generate a list of slice positions
    slices = []
    for filename in filelist:
        dataset = None
        try:
            dataset = pydicom.dcmread(filename, specific_tags=None)
        except pydicom.errors.InvalidDicomError:
            print("Error: Invalid DICOM file: " + path)
            return None

        try:
            rescaleIntercept = dataset['00281052'].value  # RescaleIntercept (b)
            rescaleSlope = dataset['00281053'].value  # RescaleSlope     (m)  U = m*SV + b
        except KeyError:
            rescaleIntercept = 0
            rescaleSlope = 1

        sl = {
            'position': np.array(dataset['00200032'].value),  # ImagePositionPatient
            'orientation': np.array(dataset['00200037'].value),  # ImageOrientationPatient
            'spacing': np.array(dataset['00280030'].value),  # PixelSpacing
            'sliceThickness': dataset['00180050'].value,  # SliceThickness
            'rows': dataset['00280010'].value,  # Rows
            'columns': dataset['00280011'].value,  # Columns
            'sliceLocation': dataset['00201041'].value,  # SliceLocation
            'bitsAllocated': dataset['00280100'].value,  # BitsAllocated
            'instanceNumber': dataset['00200013'].value,  # InstanceNumber -- image number
            'img_type': dataset['0043102F'].value  # 0 for magnitude img, 1 for phase img
        }
        if rescaleIntercept >= 0:
            sl['pixelArray'] = dataset.pixel_array * rescaleSlope + rescaleIntercept
        else:
            sl['pixelArray'] = (dataset.pixel_array * rescaleSlope + rescaleIntercept).astype('int16')

        slices.append(sl)

    # Sort the slices
    def keyfunc(e):
        return e['sliceLocation']

    slices.sort(key=keyfunc)

    ## Generate a 3D matrix
    # data = np.array([])
    #
    # print(slices[0]['pixelArray'].dtype)
    # data = np.atleast_3d(np.transpose(slices[0]['pixelArray']))
    # for sl in slices[1:]:
    #    data = np.append(data, np.atleast_3d(np.transpose(sl['pixelArray'])), axis=2)

    dim = np.array([slices[0]['columns'], slices[0]['rows'], len(slices)])

    # sliceSpacing = slices[1]['sliceLocation'] - slices[0]['sliceLocation']
    sliceSpacing = 1
    spacing = slices[0]['spacing']
    spacing = np.append(spacing, sliceSpacing)

    # Image orientation matrix
    # The orientation matrix is defined as N = [n1, n2, n3], where n1, n2, and n3 are
    # normal column vectors representing the directions of the i, j, and k indicies.
    norm = slices[0]['orientation'].reshape((2, 3))
    norm = np.append(norm, np.cross(norm[0], norm[1]).reshape((1, 3)), axis=0)
    norm = np.transpose(norm)

    # Switch from LPS to RAS
    lpsToRas = np.transpose(np.array([[-1., -1., 1.]]))
    norm = norm * lpsToRas

    # Location of the first voxel in RAS
    pos = slices[0]['position'].reshape(3, 1) * lpsToRas

    # Location of the the image center
    # OpenIGTLink uses the location of the volume center while SliceLocation in DICOM is the position of the first voxel.
    offset = norm * (spacing * (dim - 1.0) / 2.0)
    pos = pos + offset[:, [0]] + offset[:, [1]] + offset[:, [2]]

    nbytes = slices[0]['bitsAllocated'] / 8
    typestr = str(slices[0]['pixelArray'].dtype)

    # seriesDescription       = dataset['0008103e'].value # (0008,103e) : SeriesDescription
    # patientsName            = dataset['00100010'].value # (0010,0010) : PatientsName
    # studyID                 = dataset['00200010'].value # (0020,0010) : StudyID
    seriesNumber = dataset['00200011'].value  # (0020,0011) : SeriesNumber
    # imageOrientationPatient = dataset['00200037'].value # (0020,0037) : ImageOrientationPatient
    # acquisitionTime         = dataset['00080032'].value # (0008,0032) : AcquisitionTime

    ## Create an OpenIGTLink message
    DataTypeTable = {
        'int8': [2, 1],  # TYPE_INT8    = 2, 1 byte
        'uint8': [3, 1],  # TYPE_UINT8   = 3, 1 byte
        'int16': [4, 2],  # TYPE_INT16   = 4, 2 bytes
        'uint16': [5, 2],  # TYPE_UINT16  = 5, 2 bytes
        'int32': [6, 4],  # TYPE_INT32   = 6, 4 bytes
        'uint32': [7, 4],  # TYPE_UINT32  = 7, 4 bytes
        'float32': [10, 4],  # TYPE_FLOAT32 = 10, 4 bytes
        'float64': [11, 8],  # TYPE_FLOAT64 = 11, 8 bytes
    }

    imageMsg = igtl.ImageMessage.New()
    # imageMsg.SetDimensions(data.shape[0], data.shape[1], data.shape[2])
    imageMsg.SetDimensions(int(dim[0]), int(dim[1]), int(dim[2]))

    typeid = 0
    if typestr in DataTypeTable:
        typeid = DataTypeTable[typestr][0]
    else:
        print('Data type: %s is not compatible with OpenIGTLink.' % str(data.dtype))
        return

    imageMsg.SetScalarType(typeid)
    if imgname == None:
        imgname = 'IGTL_%d' % seriesNumber

    if sl['img_type'] == 0:
        imgname = imgname + "_mag"
    elif sl['img_type'] == 1:
        imgname = imgname + "_phase"

    imageMsg.SetDeviceName(imgname)
    imageMsg.SetNumComponents(1)
    imageMsg.SetEndian(2)  # little is 2, big is 1
    imageMsg.SetSpacing(spacing[0], spacing[1], spacing[2])
    imageMsg.AllocateScalars()

    matrixNP = np.concatenate((norm, pos), axis=1)
    matrixNP = np.concatenate((matrixNP, np.array([[0, 0, 0, 1]])), axis=0)
    matrix = matrixNP.tolist()

    imageMsg.SetMatrix(matrix)

    # Copy the binary data
    offset = 0
    for sl in slices:
        igtl.copyBytesToPointer(sl['pixelArray'].tobytes(), igtl.offsetPointer(imageMsg.GetScalarPointer(), offset))
        offset = offset + len(sl['pixelArray'].tobytes())

    imageMsg.Pack()

    sock.Send(imageMsg.GetPackPointer(), imageMsg.GetPackSize())


def groupBySeriesAndExport(cur, tags, valueListDict, cond=None, imgname=None, sock=None):
    if len(tags) == 0:
        cur.execute('SELECT path FROM dicom WHERE ' + cond)
        paths = cur.fetchall()
        if len(paths) == 0:
            return
        filelist = []
        for p in paths:
            filelist.append(str(p[0]))

        exportToIGTL(filelist, sock=sock, imgname=imgname)

        return

    # Note: We add prefix 'x' to the DICOM tag as the DICOM tags are recognized as intenger
    #       by SQLight
    tag = 'x' + tags[0].replace(',', '')
    values = list(valueListDict[tag])
    tags2 = tags[1:]

    for tp in values:
        value = tp[0]
        cond2 = ''
        imgname2 = ''
        if cond == None:
            cond2 = tag + ' == ' + "'" + value + "'"
        else:
            cond2 = cond + ' AND ' + tag + ' == ' + "'" + value + "'"
        if imgname == None:
            imgname2 = 'IGTL-' + value
        else:
            imgname2 = imgname + '-' + value
        groupBySeriesAndExport(cur, tags2, valueListDict, cond2, imgname2, sock=sock)


def main(argv):
    try:
        parser = argparse.ArgumentParser(description="Split DICOM series by Tag.")
        parser.add_argument('tags', metavar='TAG', type=str, nargs='+',
                            help='DICOM tags(e.g. "0020,000E")')
        parser.add_argument('src', metavar='SRC_DIR', type=str, nargs=1,
                            help='source directory')
        parser.add_argument('ip', metavar='IP', type=str, nargs=1,
                            help='IP address of the destination host')
        parser.add_argument('port', metavar='PORT', type=str, nargs=1,
                            help='Port number of the destination host')
        parser.add_argument('-r', dest='recursive', action='store_const',
                            const=True, default=False,
                            help='search the source directory recursively')
        args = parser.parse_args(argv)

    except Exception as e:
        print(e)

    tags = args.tags
    srcdir = args.src[0]
    ip = args.ip[0]
    port = args.port[0]

    con = sqlite3.connect(':memory:')
    # con = sqlite3.connect('TestDB.db')
    cur = con.cursor()

    buildFilePathDBByTags(con, srcdir, tags, True)

    # Generate a list of values for each tag
    valueListDict = {}
    for tag in tags:
        colName = 'x' + tag.replace(',', '')
        cur.execute('SELECT ' + colName + ' FROM dicom GROUP BY ' + colName)
        valueListDict[colName] = cur.fetchall()

    clientSocket = igtl.ClientSocket.New()
    clientSocket.SetReceiveTimeout(1)  # Milliseconds
    ret = clientSocket.ConnectToServer(ip, int(port))
    if ret == 0:
        print('Connection successful.')
        groupBySeriesAndExport(cur, tags, valueListDict, cond=None, imgname=None, sock=clientSocket)
    else:
        print('Could not connect to the server.')

    sys.exit()


if __name__ == "__main__":
    # main(sys.argv[1:])
    pass

