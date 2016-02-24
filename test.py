import cv2
import numpy as np

from NLMeans3D import *

nlm = NLMeans3D()

videoFile = '/home/tiagosn/datasets_sibgrapi2016/KTH_Actions/sequences/test/boxing/person02_boxing_d1_uncomp-frames-1-105.avi'
cap = cv2.VideoCapture(videoFile)
gray = None

nFrames = int(cap.get(cv2.cv.CV_CAP_PROP_FRAME_COUNT))
print 'nframes = %d' % (nFrames)

i = 0
while(cap.isOpened()):
    ret, frame = cap.read()

    if frame is None:
        break

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    gray = gray.astype(np.float64)/255.0

    if i == 0:
        video = np.zeros((nFrames, gray.shape[0], gray.shape[1]))

    video[i] = gray

    #cv2.imshow('frame', video[i])
    #cv2.waitKey(30)

    i += 1

out = nlm.denoise(video)

width  = int(cap.get(cv2.cv.CV_CAP_PROP_FRAME_WIDTH))
height = int(cap.get(cv2.cv.CV_CAP_PROP_FRAME_HEIGHT))
fps = cap.get(cv2.cv.CV_CAP_PROP_FPS)
fourcc = cv2.cv.CV_FOURCC('M', 'J', 'P', 'G')
writer = cv2.VideoWriter("/home/tiagosn/Desktop/videoOriginal.avi",fourcc,fps,(width,height), False)

for i in range(0, nFrames):
    cv2.imshow('frame', video[i])
    cv2.waitKey(30)
    aux = out[i] * 255
    aux = aux.astype(np.uint8)
    writer.write(aux)

writer.release()

print '-----------------------'
