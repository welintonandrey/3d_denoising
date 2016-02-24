import os
import cv2
import numpy as np

pathOriginalFrames = '/home/tiagosn/Desktop/3D-NL-Means/videos/seq1/' # colocar ser o / no final
std = 25

newPath = pathOriginalFrames + 'gaussian_noise-' + str(std)
if not os.path.exists(newPath):
    os.makedirs(newPath)

frameList = sorted(os.listdir(pathOriginalFrames))
frameList = [f for f in frameList if '.png' in f]
for f in frameList:
    originalImagePath = pathOriginalFrames + f
    original = cv2.imread(originalImagePath)
    gray = cv2.cvtColor(original, cv2.COLOR_BGR2GRAY)
    gray = gray.astype(np.float64)#/255.0

    noise = np.random.normal(0, std, gray.shape)
    noiseImage = gray + noise

    noiseImage[noiseImage > 255.0] = 255.0
    noiseImage[noiseImage < 0.0] = 0.0
    noiseImage = noiseImage.astype(np.uint8)

    noiseImagePath = newPath + '/' + 'gn-' + str(std) + '-' + f
    cv2.imwrite(noiseImagePath, noiseImage)

#noise = np.random.normal(0,1,(10,10))