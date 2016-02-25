import cv2
import sys
import numpy as np
from  scipy import ndimage

import itertools
import joblib
from joblib import Parallel, delayed

def processPixel(video, t, i, j, h, halfWindowSize, halfTemplate, gaussian):
    ws = 2*halfWindowSize+1
    w = np.zeros((ws, ws, ws))

    pc = video[t - halfTemplate: t + halfTemplate + 1, \
                i - halfTemplate: i + halfTemplate + 1, \
                j - halfTemplate: j + halfTemplate + 1]

    for tt in xrange(t - halfWindowSize, t + halfWindowSize + 1):
        for ii in xrange(i - halfWindowSize, i + halfWindowSize + 1):
            for jj in xrange(j - halfWindowSize, j + halfWindowSize + 1):
                pn = video[tt - halfTemplate: tt + halfTemplate + 1, \
                            ii - halfTemplate: ii + halfTemplate + 1, \
                            jj - halfTemplate: jj + halfTemplate + 1]

                tw = tt - t + halfWindowSize
                iw = ii - i + halfWindowSize
                jw = jj - j + halfWindowSize

                dist2 = np.sum(((pc-pn)**2)*gaussian)
                w[tw, iw, jw] = np.exp(-(dist2/(h*h)))

    w[halfWindowSize, halfWindowSize, halfWindowSize] = 0.0
    w[halfWindowSize, halfWindowSize, halfWindowSize] = np.max(w)

    w = w / np.sum(w)
    neighborhood = video[t - halfWindowSize: t + halfWindowSize + 1, \
                        i - halfWindowSize: i + halfWindowSize + 1, \
                        j - halfWindowSize: j + halfWindowSize + 1]

    print 'Pixel (%3d, %3d) processed!!! ' % (i, j)
    return np.sum(w*neighborhood)

class ParNLMeans3D:
    def __init__(self, h = 3, templateWindowSize = 7, searchWindowSize = 21, sigma = 1):
        self.h = h
        self.templateWindowSize = templateWindowSize
        self.searchWindowSize = searchWindowSize
        self.sigma = sigma

    # video -> numpy array (3D)
    def denoise(self, video):
        out = video.copy()
        #out = np.ones(video.shape)

        nFrames = video.shape[0]
        nRows = video.shape[1]
        nCols = video.shape[2]

        halfWindowSize = self.searchWindowSize / 2
        halfTemplate = self.templateWindowSize / 2
        delta = halfWindowSize + halfTemplate

        aux = np.zeros((self.templateWindowSize, self.templateWindowSize, self.templateWindowSize))
        aux[halfTemplate, halfTemplate, halfTemplate] = 1
        gaussian = ndimage.filters.gaussian_filter(aux, self.sigma)

        ranges = [range(delta, nFrames - delta), range(delta, nRows - delta), range(delta, nCols - delta)]
        coordinates = list(itertools.product(*ranges))

        ncpus = joblib.cpu_count()
        results = Parallel(n_jobs=ncpus,max_nbytes=2e9)(delayed(processPixel)(video, t, i, j, self.h, halfWindowSize, halfTemplate, gaussian) for t,i,j in coordinates)

        for idx in xrange(0,len(results)):
            out[coordinates[idx][0], coordinates[idx][1], coordinates[idx][2]] = results[idx]

        return out
