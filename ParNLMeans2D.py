import cv2
import sys
import numpy as np
from  scipy import ndimage
import time
import itertools
import joblib
from joblib import Parallel, delayed
from bob.ip.base import *

from printProgressBar import *

def processPixel(image, i, j, h, halfWindowSize, halfTemplate, gaussian):
    delta = halfWindowSize + halfTemplate
    ws = 2*halfWindowSize+1
    w = np.zeros((ws, ws))
    w_texLBP = np.zeros((ws, ws))

    pc = image[i - halfTemplate: i + halfTemplate + 1, \
                j - halfTemplate: j + halfTemplate + 1]

    lbp = bob.ip.base.LBP(8, uniform=True, rotation_invariant=True);
    histc = np.histogram(lbp.extract(pc), lbp.max_label, (0,lbp.max_label))[0]

    for ii in xrange(i - halfWindowSize, i + halfWindowSize + 1):
        for jj in xrange(j - halfWindowSize, j + halfWindowSize + 1):
            pn = image[ii - halfTemplate: ii + halfTemplate + 1, \
                        jj - halfTemplate: jj + halfTemplate + 1]

            iw = ii - i + halfWindowSize
            jw = jj - j + halfWindowSize

            histn = np.histogram(lbp.extract(pn), lbp.max_label, (0,lbp.max_label))[0]
            w_texLBP[iw, jw] =  np.sum(((histc - histn)**2) / (histc + histn))

            dist2 = np.sum(((pc-pn)**2)*gaussian)
            w[iw, jw] = np.exp(-(dist2/(h*h)))


    # calc w_Intensity
    w[halfWindowSize, halfWindowSize] = 0.0
    w[halfWindowSize, halfWindowSize] = np.max(w)
    w = w / np.sum(w)

    # Normalize weights matrix (texture = LBP)
    # Aux (Not returned)
    hs = np.std(w_texLBP)
    w_texLBP = np.exp(-w_texLBP/hs)
    w_texLBP[halfWindowSize, halfWindowSize, halfWindowSize] = 0.0
    w_texLBP[halfWindowSize, halfWindowSize, halfWindowSize] = np.max(w_texLBP)
    w_texLBP = w_texLBP / np.sum(w_texLBP)

    # Calc and Normalize weights matrix (texture = LBP)
    w_NLM_LBP = w * w_texLBP
    w_NLM_LBP = w_NLM_LBP / np.sum(w_NLM_LBP)

    neighborhood = image[i - halfWindowSize: i + halfWindowSize + 1, \
                        j - halfWindowSize: j + halfWindowSize + 1]

    totalPixel = (image.shape[0] - 2*delta) * (image.shape[1] - 2*delta)
    auxP = ((i-delta) * (image.shape[1]-2*delta) + (j-delta+1))
    printProgressBar(auxP, totalPixel)

    #Return result
    return np.sum(w*neighborhood), np.sum(w_NLM_LBP*neighborhood)

class ParNLMeans2D:
    def __init__(self, h = 3, templateWindowSize = 7, searchWindowSize = 21, sigma = 1):
        self.h = h
        self.templateWindowSize = templateWindowSize
        self.searchWindowSize = searchWindowSize
        self.sigma = sigma

    # image -> numpy array (2D)
    def denoise(self, imageIn):
        halfWindowSize = self.searchWindowSize / 2
        halfTemplate = self.templateWindowSize / 2
        delta = halfWindowSize + halfTemplate


        shape = tuple(np.add(imageIn.shape, (2*delta, 2*delta)))
        image = np.zeros(shape)

        image = cv2.copyMakeBorder(imageIn, delta, delta, delta, delta, cv2.BORDER_REFLECT_101)

        out = image.copy()
        outLBP = image.copy()

        nRows = image.shape[0]
        nCols = image.shape[1]

        aux = np.zeros((self.templateWindowSize, self.templateWindowSize))
        aux[halfTemplate, halfTemplate] = 1
        gaussian = ndimage.filters.gaussian_filter(aux, self.sigma)

        ranges = [range(delta, nRows - delta), range(delta, nCols - delta)]
        coordinates = list(itertools.product(*ranges))

        ncpus = joblib.cpu_count()
        results = Parallel(n_jobs=ncpus,max_nbytes=2e9)(delayed(processPixel)(image, i, j, self.h, halfWindowSize, halfTemplate, gaussian) for i,j in coordinates)
        printProgressBar(100, 100)

        for idx in xrange(0,len(results)):
            out[coordinates[idx][0], coordinates[idx][1]] = results[idx][0]
            outLBP[coordinates[idx][0], coordinates[idx][1]] = results[idx][1]

        return out[delta: -delta, delta: -delta], \
                outLBP[delta: -delta, delta: -delta]
