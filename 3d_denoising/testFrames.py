import os
import cv2
import numpy as np
from NLMeans3D import *

pathNoiseFrames = '/home/tiagosn/Desktop/3D-NL-Means/videos/seq2/25'

newPath = pathNoiseFrames + '-denoise-nlm3d_t8'
if not os.path.exists(newPath):
    os.makedirs(newPath)


frameList = sorted(os.listdir(pathNoiseFrames))
frameList = [f for f in frameList if '.png' in f]

nFrames = len(frameList)
i = 0

nlm = NLMeans3D(20,5,9)
for f in frameList:
    originalImagePath = pathNoiseFrames + '/' + f
    print originalImagePath
    original = cv2.imread(originalImagePath)
    gray = cv2.cvtColor(original, cv2.COLOR_BGR2GRAY)
    gray = gray.astype(np.float64)#/255.0

    if i == 0:
        video = np.zeros((nFrames, gray.shape[0], gray.shape[1]))

    video[i] = gray
    i += 1

print video.shape
out = nlm.denoise(video)

for i in range(0, nFrames):
    cv2.imshow('frame', video[i])
    cv2.waitKey(30)
    aux = out[i]# * 255
    aux = aux.astype(np.uint8)

    denoiseImagePath = newPath + '/' + 'denoise-' + str(i) + '.png'
    cv2.imwrite(denoiseImagePath, aux)
