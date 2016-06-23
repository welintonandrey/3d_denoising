import os
import cv2
import numpy as np

import sys, argparse

parser = argparse.ArgumentParser(description='Script for insert noise in images.')
parser.add_argument('-in','--input', help='Input PATH with the input images (Color images)',required=True)
parser.add_argument('-sig','--sigma',help='Input the parameter Sigma (Noise value)', type=int, required=False)

args = parser.parse_args()

pathOriginalFrames = args.input
std = args.sigma

newPath = pathOriginalFrames + 'gaussian_noise-' + str(std)
if not os.path.exists(newPath):
    os.makedirs(newPath)

frameList = sorted(os.listdir(pathOriginalFrames))
frameList = [f for f in frameList if '.png' in f]
for f in frameList:
    originalImagePath = pathOriginalFrames + f
    original = cv2.imread(originalImagePath, cv2.IMREAD_GRAYSCALE)
    gray = original.astype(np.float64)#/255.0

    if std == 0:
        noiseImage = gray
    else:
        noise = np.random.normal(0, std, gray.shape)
        noiseImage = gray + noise

    noiseImage[noiseImage > 255.0] = 255.0
    noiseImage[noiseImage < 0.0] = 0.0
    noiseImage = noiseImage.astype(np.uint8)

    noiseImagePath = newPath + '/' + 'gn-' + str(std) + '-' + f
    cv2.imwrite(noiseImagePath, noiseImage)

#noise = np.random.normal(0,1,(10,10))
