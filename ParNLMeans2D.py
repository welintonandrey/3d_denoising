import cv2
import sys
import numpy as np
from  scipy import ndimage
import time
import itertools
import joblib
from joblib import Parallel, delayed

def processPixel(image, i, j, h, halfWindowSize, halfTemplate, gaussian):
    delta = halfWindowSize + halfTemplate
    ws = 2*halfWindowSize+1
    w = np.zeros((ws, ws))

    pc = image[i - halfTemplate: i + halfTemplate + 1, \
                j - halfTemplate: j + halfTemplate + 1]

    for ii in xrange(i - halfWindowSize, i + halfWindowSize + 1):
        for jj in xrange(j - halfWindowSize, j + halfWindowSize + 1):
            pn = image[ii - halfTemplate: ii + halfTemplate + 1, \
                        jj - halfTemplate: jj + halfTemplate + 1]

            iw = ii - i + halfWindowSize
            jw = jj - j + halfWindowSize

            dist2 = np.sum(((pc-pn)**2)*gaussian)
            w[iw, jw] = np.exp(-(dist2/(h*h)))


    # calc w_Intensity
    w[halfWindowSize, halfWindowSize] = 0.0
    w[halfWindowSize, halfWindowSize] = np.max(w)
    w = w / np.sum(w)

    neighborhood = image[i - halfWindowSize: i + halfWindowSize + 1, \
                        j - halfWindowSize: j + halfWindowSize + 1]

    #print 'Pixel (%3d, %3d) processed!!! ' % (i-delta, j-delta)
    totalPixel = (image.shape[0] - 2*delta) * (image.shape[1] - 2*delta)
    auxP = ((i-delta) * image.shape[1] + (j-delta))
    #print image.shape
    auxP = ((100 * auxP) / totalPixel)
    #print ("%d de %d") % (auxP, totalPixel)
    auxP = 0 if auxP<0 else 100 if i>100 else auxP
    sys.stdout.write('\rProcessando: %3d%%' % auxP)
    sys.stdout.flush()
    #exit()

    #Return result
    return np.sum(w*neighborhood)

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

        nRows = image.shape[0]
        nCols = image.shape[1]

        aux = np.zeros((self.templateWindowSize, self.templateWindowSize))
        aux[halfTemplate, halfTemplate] = 1
        gaussian = ndimage.filters.gaussian_filter(aux, self.sigma)

        ranges = [range(delta, nRows - delta), range(delta, nCols - delta)]
        coordinates = list(itertools.product(*ranges))

        ncpus = joblib.cpu_count()
        results = Parallel(n_jobs=ncpus,max_nbytes=2e9)(delayed(processPixel)(image, i, j, self.h, halfWindowSize, halfTemplate, gaussian) for i,j in coordinates)

        for idx in xrange(0,len(results)):
            out[coordinates[idx][0], coordinates[idx][1]] = results[idx]

        return out[delta: -delta, delta: -delta]
