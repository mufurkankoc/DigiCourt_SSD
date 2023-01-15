from array import array
from itertools import count
import imutils
from sqlite3 import converters
import threading
import jetson.inference
import jetson.utils
import cv2
from collections import deque
from imutils.video import VideoStream
from pypylon import pylon
import time
import matplotlib.pyplot as plt
import numpy as np
from scipy.signal import find_peaks


# greenLower = (29, 86, 6)
# greenUpper = (64, 255, 255)

# conecting to the first available camera
camera = pylon.InstantCamera(pylon.TlFactory.GetInstance().CreateFirstDevice())
cap = cv2.VideoCapture(0)
cap.set(3, 640)
cap.set(4, 480)

# # # # # # # # # # # # # ## # # # # # # # # # # # # # # # # # # # # # # # # # # # # #

plotlistesi=[]
plotlistesi2=[]
totallist=[]
totallist2=[]
bouncepointlist=[]
bouncepointlist2=[]

trajcross=[]
lowercenter=[]
trajlist=[]
baspeaks =[]
lowercenter2=[]
trajlist2=[]
# # # # # # # # # # # # # ## # # # # # # # # # # # # # # # # # # # # # # # # # # # # #

img5 = np.zeros ((480,640,1))
img6 = np.zeros ((480,640,1))
img7 = np.zeros ((480,640,1))

trajcross2=[]


# Grabing Continusely (video) with minimal delay
camera.StartGrabbing(pylon.GrabStrategy_LatestImageOnly)
camera.ExposureTime.SetValue(12000.0)
camera.BalanceWhiteAuto.SetValue("Continuous")
converter = pylon.ImageFormatConverter()
# converting to opencv bgr format
converter.OutputPixelFormat = pylon.PixelType_BGR8packed
converter.OutputBitAlignment = pylon.OutputBitAlignment_MsbAligned

# # # # # # # # # # # köşeler# # # # # #
center_coordinates1 = (240, 226)
center_coordinates2 = (156, 378)
center_coordinates3 = (450, 389)
center_coordinates4 = (428, 233)
#radius = 2
color = (0, 0, 255)
thickness = -1
# # # # # # # # # # # # # ## # # # # # # # # # # # # # # # # # # # # # # # # # # # # #

ortalamax=[]
ortalamay=[]



# # # # # # # # # # PolyLines# # # # # ####### # # # # # # # # # # # # # # # # # # # # # #

pts = np.array([[3, 451], [567, 450], [559, 146], [204, 248]], np.int32)

pts = pts.reshape((-1, 1, 2))
isClosed = True
color2 = (255, 0, 0)
thickness2 = 2
# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #

pts2 = np.array([[322, 4], [348, 475], [638, 477], [637, 3]], np.int32)

pts2 = pts2.reshape((-1, 1, 2))
isClosed = True
color2 = (255, 0, 0)
thickness2 = 2


posListX = []
posListY = []


xList = [item for item in range(0, 640)]
yList = [item for item in range(0, 480)]

# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #

posListX2 = []
posListY2 = []

xList2 = [item for item in range(0, 640)]
yList2 = [item for item in range(0, 480)]


# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #

farklistesiic = []
farklistesidis = []

webcamic = []
webcamdis = []

# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #

fps_count = 0
fps_count2 = 0


# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #

def degerlendirme(deger):
    global farklistesiic
    global farklistesidis

    x1 = int(deger[0])

    y1 = int(deger[1])

    # # # # # # # # # # PolyLines# # # # # ####### # # # # # # # # # # # # # # # # # # # # # #

    pts = np.array([[[3, 451], [567, 450], [559, 146], [204, 248]]], np.int32)

    pts = pts.reshape((-1, 1, 2))
    isClosed = True
    color2 = (255, 0, 0)
    thickness2 = 2

    # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #

    dist = cv2.pointPolygonTest(pts, (x1, y1), False)
    if dist == 1.0:
        farklistesiic.append(deger)
    elif dist == -1.0:
        farklistesidis.append(deger)

    print(deger)

# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #

# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #


def webdegerlendirme(deger):
    global webcamic
    global webcamdis

    x1 = int(deger[0])

    y1 = int(deger[1])

    # # # # # # # # # # PolyLines# # # # # ####### # # # # # # # # # # # # # # # # # # # # # #

    pts = np.array([[322, 4], [348, 475], [638, 477], [637, 3]], np.int32)

    pts = pts.reshape((-1, 1, 2))
    isClosed = True
    color2 = (255, 0, 0)
    thickness2 = 2

    # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #

    dist = cv2.pointPolygonTest(pts, (x1, y1), False)
    if dist == 1.0:
        webcamic.append(deger)
    elif dist == -1.0:
        webcamdis.append(deger)

    print(deger)

# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #

def minima(plotlistesi):
    xxx = np.array(plotlistesi)
    yyy = xxx*np.random.randn(len(xxx))**2
    peaks = find_peaks(yyy, height = 1, threshold = 1, distance = 1)
    height = peaks[1]['peak_heights'] #list containing the height of the peaks
    
    roundlist=np.round(yyy).astype(int)
    heightmax=int(np.round(max(height)))
    linevalue=int(np.where(heightmax==roundlist)[0])
    
    peak_pos = xxx[peaks[0]] 
    y2 = yyy*-1
    minima = find_peaks(y2, threshold = 1, distance = 1)
    min_pos = xxx[minima[0]]   #list containing the positions of the minima
    min_height = y2[minima[0]]   #list containing the height of the minima
    fig = plt.figure()
    ax = fig.subplots()
    ax.plot(xxx,yyy)
    ax.scatter(peak_pos, height, color = 'r', s = 10, marker = 'D', label = 'maxima')
    ax.scatter(min_pos, min_height*-1, color = 'gold', s = 10, marker = 'X', label = 'minima')
    ax.legend()
    ax.grid()
    # plt.show()


    return linevalue




def minima2(plotlistesi):
    xxx = np.array(plotlistesi)
    yyy = xxx*np.random.randn(len(xxx))**2
    peaks = find_peaks(yyy, height = 1, threshold = 1, distance = 1)
    height = peaks[1]['peak_heights'] #list containing the height of the peaks
    
    roundlist=np.round(yyy).astype(int)
    heightmax=int(np.round(max(height)))
    linevalue=int(np.where(heightmax==roundlist)[0])
    
    peak_pos = xxx[peaks[0]] 
    y2 = yyy*-1
    minima = find_peaks(y2, threshold = 1, distance = 1)
    min_pos = xxx[minima[0]]   #list containing the positions of the minima
    min_height = y2[minima[0]]   #list containing the height of the minima
    fig = plt.figure()
    ax = fig.subplots()
    ax.plot(xxx,yyy)
    ax.scatter(peak_pos, height, color = 'r', s = 10, marker = 'D', label = 'maxima')
    ax.scatter(min_pos, min_height*-1, color = 'gold', s = 10, marker = 'X', label = 'minima')
    ax.legend()
    ax.grid()
    # plt.show()


    return linevalue


def basler():

    try:
        net = jetson.inference.detectNet('ssd-mobilenet-v2', threshold=0.05)
    except:
        pass


    global fps_count2
    crashesbas =[]
    img5 = np.zeros ((480,640,1))
    img6 = np.zeros ((480,640,1))
    img7 = np.zeros ((480,640,1))
    
    while True:

        fps_count2 += 1

        grabResult = camera.RetrieveResult(
            5000, pylon.TimeoutHandling_ThrowException)
        image = converter.Convert(grabResult)
        img = image.GetArray()
        img = cv2.resize(img, (640, 480))
        img = cv2.polylines(img, [pts], isClosed, color2, thickness2)
        imgCuda = jetson.utils.cudaFromNumpy(img)
        detections = net.Detect(imgCuda)
        # cv2.putText(img, f'FPS: {int(net.GetNetworkFPS())}', (30, 30),
        #             cv2.FONT_HERSHEY_DUPLEX, 1, (255, 0, 0), 2)

        for d in detections:

            className = net.GetClassDesc(d.ClassID)

            if className == "sports ball" or className == "apple":

                fps_count2 = 0
                x1, y1, x2, y2 = int(d.Left), int(
                    d.Top), int(d.Right), int(d.Bottom)
                ortayol = int((x2+x1)/2)
                alt = (ortayol,  y2)
                cx,cy=int(d.Center[0]),int(d.Center[1])
                center=[int(cx),int(cy)]

                posListX2.append(ortayol)
                posListY2.append(y2)
                plotlistesi2.append(y2)
                totallist2.append([int(x2),int(y2)])

                cv2.circle(img, alt, 5, (200, 50, 75), cv2.FILLED)
                cv2.rectangle(img, (x1, y1), (x2, y2), (255, 0, 255), 2)
                cv2.putText(img, className, (x1+5, y1+15),
                            cv2.FONT_HERSHEY_DUPLEX, 0.75, (255, 0, 255), 2)

        #     print(pospeaks)
        if fps_count2 > 10 and len(posListY2) > 3:

            try:

                max_value = np.max(posListY2)
                result = np.where(posListY2 == max_value)
                result = int(result[0])

                if posListY2[result+1] <posListY2[result+2]:
                    result = result+1

                arrU_Y = posListY2[0:result+1]
                arrD_Y = posListY2[result+1:]

                arrU_X = posListX2[0:result+1]
                arrD_X = posListX2[result+1:]

                result3 = posListX2[result]

                toplok = ([result3, max_value])
                degerlendirme(toplok)
                cv2.circle(img, (result3, max_value), 10,
                           (100, 10, 100), cv2.FILLED)
                # print(":",int((posListY2[result] + posListY2[result+1])/2))
                # print(":", int((posListX2[result] + posListX2[result+1])/2))
                crashY =int((posListY2[result] + posListY2[result+1])/2)
                crashX =int((posListX2[result] + posListX2[result+1])/2)
                crash =[crashX, crashY]
                crashesbas.append(crash)




            except:
                pass

            try:
                A, B, C = np.polyfit(arrD_X, arrD_Y, 2)

                for x in xList2:

                    c = int(A * x ** 2 + B * x + C)
                    cv2.circle(img, (x, c), 5, (255, 0, 255), cv2.FILLED)
                    cv2.circle(img5, (x,c), 5, (255,255,255), cv2.FILLED)
            except:
                continue

            try:
                A, B, C = np.polyfit(arrU_X, arrU_Y, 2)

                for x in xList2:

                    y = int(A * x ** 2 + B * x + C)
                    cv2.circle(img, (x, y), 5, (100, 50, 75), cv2.FILLED)
                    cv2.circle(img6, (x,y), 5, (255,255,255), cv2.FILLED)


            except:
                continue

            numberofpoint2=minima2(plotlistesi2)
            bouncepoint2=totallist2[numberofpoint2]
            # print("bouncepoint "+ bouncepoint)
            bouncepointlist2.append(bouncepoint2)


            img7 = cv2.bitwise_and(img5,img6)
            img7=img7.astype(np.uint8)
            cnts = cv2.findContours(img7.copy(), cv2.RETR_EXTERNAL,
        cv2.CHAIN_APPROX_SIMPLE)
            cnts = imutils.grab_contours(cnts)
            gbx =0
            print(len(cnts),"len")
            centers=[]
            if len(cnts)>0:
                for i in cnts:

                    c1 = cv2.moments(cnts[gbx])
                    if c1["m00"] != 0:
                        cX = int(c1["m10"] / c1["m00"])
                        cY = int(c1["m01"] / c1["m00"])
                    else:
                        # set values as what you need in the situation
                        cX, cY = 0, 0
                    center1 = (cX, cY)
                    centers.append(center1)
                    gbx += 1
                location = (np.argmax(centers, axis=0))[1]
                lowercenter2.append(centers[location])
            if len(centers) ==0 and toplok and crash:
                norm = ((int((toplok[0] + crash[0])/2), int((toplok[1]+ crash[1])/2)))
                lowercenter2.append(norm)
                print("ortalama")
            elif len(centers) ==0 and toplok:
                lowercenter2.append(toplok)
                print("toplok")
                print(centers,"")


            print(centers ,'centers')

            cv2.circle(img, center, 5, (0, 0, 255), -1)
            trajlist2.append(center)
            img5 = np.zeros ((480,640,1))
            img6 = np.zeros ((480,640,1))

            posListX2.clear()
            posListY2.clear()
            plotlistesi2.clear()
            totallist2.clear()

        for i in farklistesiic:
            cv2.circle(img, (i[0], i[1]), 5, (0, 255, 0), cv2.FILLED)

        for i in farklistesidis:
            cv2.circle(img, (i[0], i[1]), 5, (0, 0, 255), cv2.FILLED)

        for i in bouncepointlist2:
            cv2.circle(img, (i[0], i[1]), 5, (255, 0, 0), cv2.FILLED)    

        cv2.imshow("Image", img)
        cv2.waitKey(2)


def webcam():
    try:
        net = jetson.inference.detectNet('ssd-mobilenet-v2', threshold=0.05)
    except:
        pass

    global fps_count
    crashesweb =[]
    img2 = np.zeros ((480,640,1))
    img3 = np.zeros ((480,640,1))
    img4 = np.zeros ((480,640,1))

    while True:

        fps_count += 1

        success, img = cap.read()
        # img = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
        img = cv2.polylines(img, [pts2], isClosed, color2, thickness2)
        imgCuda = jetson.utils.cudaFromNumpy(img)
        detections = net.Detect(imgCuda)

        for d in detections:

            className = net.GetClassDesc(d.ClassID)

            if className == "sports ball" or className == "apple":

                fps_count = 0
                x1, y1, x2, y2 = int(d.Left), int(
                    d.Top), int(d.Right), int(d.Bottom)
                ortayol = int((x2+x1)/2)
                alt = (ortayol,  y2)
                cx,cy=int(d.Center[0]),int(d.Center[1])
                center=[int(cx),int(cy)]

                posListX.append(ortayol)
                posListY.append(y2)
                plotlistesi.append(y2)
                totallist.append([int(x2),int(y2)])


                cv2.circle(img, alt, 5, (200, 50, 75), cv2.FILLED)
                cv2.rectangle(img, (x1, y1), (x2, y2), (255, 0, 255), 2)
                cv2.putText(img, className, (x1+5, y1+15),
                            cv2.FONT_HERSHEY_DUPLEX, 0.75, (255, 0, 255), 2)

        if fps_count > 10 and len(posListY) > 3:

            try:

                max_value = np.max(posListY)
                result = np.where(posListY == max_value)
                result = int(result[0])

                arrU_Y = posListY[0:result+1]
                arrD_Y = posListY[result+1:]

                arrU_X = posListX[0:result+1]
                arrD_X = posListX[result+1:]

                result3 = posListX[result]

                toplok = ([result3, max_value])
                webdegerlendirme(toplok)

                crashY =int((posListY[result] + posListY[result+1])/2)
                crashX =int((posListX[result] + posListX[result+1])/2)
                crash =[crashX, crashY]
                crashesweb.append(crash)
            except:
                pass




            try:
                A, B, C = np.polyfit(arrD_X, arrD_Y, 2)

                for x in xList:

                    c = int(A * x ** 2 + B * x + C)
                    cv2.circle(img, (x, c), 5, (255, 0, 255), cv2.FILLED)
                    cv2.circle(img2, (x,c), 5, (255,255,255), cv2.FILLED)

            except:
                continue

            try:
                A, B, C = np.polyfit(arrU_X, arrU_Y, 2)

                for x in xList:

                    y = int(A * x ** 2 + B * x + C)
                    cv2.circle(img, (x, y), 5, (100, 50, 75), cv2.FILLED)
                    cv2.circle(img3, (x,y), 5, (255,255,255), cv2.FILLED)

            except:
                continue

            numberofpoint=minima(plotlistesi)
            bouncepoint=totallist[numberofpoint]
            # print("bouncepoint "+ bouncepoint)
            bouncepointlist.append(bouncepoint)
            print(bouncepointlist)

            img4 = cv2.bitwise_and(img2,img3)
            img4=img4.astype(np.uint8)
            cnts = cv2.findContours(img4.copy(), cv2.RETR_EXTERNAL,
        cv2.CHAIN_APPROX_SIMPLE)
            cnts = imutils.grab_contours(cnts)
            gbx =0
            print(len(cnts),"len")
            centers=[]
            if len(cnts)>0:
                for i in cnts:

                    c1 = cv2.moments(cnts[gbx])

                    if c1["m00"] != 0:
                        cX = int(c1["m10"] / c1["m00"])
                        cY = int(c1["m01"] / c1["m00"])
                    else:
                        
                        cX, cY = 0, 0
                    center1 = (cX, cY)
                    centers.append(center1)
                    gbx += 1
                location = (np.argmax(centers, axis=0))[1]
                lowercenter.append(centers[location])
            if len(centers) ==0 and toplok and crash:
                norm = ((int((toplok[0] + crash[0])/2), int((toplok[1]+ crash[1])/2)))
                lowercenter.append(norm)
                print("ortalama")
            elif len(centers) ==0 and toplok:
                lowercenter.append(toplok)
                print("toplok")
                print(centers,"")




            print(centers ,'centers')

            cv2.circle(img, center, 5, (0, 0, 255), -1)
            trajlist.append(center)
            img2 = np.zeros ((480,640,1))
            img3 = np.zeros ((480,640,1))


            # minima(posListY)
            posListX.clear()
            posListY.clear()
            plotlistesi.clear()
            totallist.clear()


        for i in webcamic:
            cv2.circle(img, (i[0], i[1]), 5, (0, 255, 0), cv2.FILLED)

        for i in webcamdis:
            cv2.circle(img, (i[0], i[1]), 5, (0, 0, 255), cv2.FILLED)

        for i in bouncepointlist:
            cv2.circle(img, (i[0], i[1]), 5, (255, 0, 0), cv2.FILLED)
        
        cv2.imshow("ImageColor", img)
        k= cv2.waitKey(2) & 0xFF
        if k == 27:    
            cv2.destroyAllWindows()        
            break
    

t1 = threading.Thread(target=basler)
t2 = threading.Thread(target=webcam)
t1.start()
t2.start()
