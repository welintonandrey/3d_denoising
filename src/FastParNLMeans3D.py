import cv2
import sys
import numpy as np
import numpy.linalg as la
from  scipy import ndimage

import itertools
import joblib
from joblib import Parallel, delayed

from LBPTOP import *
from bob.ip.base import LBP

def hist(lbpVideos, ti, tf, ri, rf, ci, cf, sizeXY, sizeXT, sizeYT):
    d = 0.0001

    hist = np.concatenate((np.histogram(lbpVideos['xy'][ti:tf,ri:rf,ci:cf], sizeXY)[0], \
                            np.histogram(lbpVideos['xt'][ti:tf,ri:rf,ci:cf], sizeXT)[0], \
                            np.histogram(lbpVideos['yt'][ti:tf,ri:rf,ci:cf], sizeYT)[0])) + d
    hist = hist / np.sum(hist)
    return hist

def processPixel(video, t, i, j, h, halfWindowSize, halfTemplate, gaussian, lbpVideos, sizeXY, sizeXT, sizeYT):
    delta = halfWindowSize + halfTemplate
    ws = 2*halfWindowSize+1
    w = np.zeros((ws, ws, ws))
    wLBP = np.zeros((ws, ws, ws))

    pc = video[t - halfTemplate: t + halfTemplate + 1, \
                i - halfTemplate: i + halfTemplate + 1, \
                j - halfTemplate: j + halfTemplate + 1]
    histc = hist(lbpVideos, \
                    t - halfTemplate, t + halfTemplate + 1, \
                    i - halfTemplate, i + halfTemplate + 1, \
                    j - halfTemplate, j + halfTemplate + 1, \
                    sizeXY, sizeXT, sizeYT)

    nonUniformPixel = (histc[9] + histc[19] + histc[29])
    nonUniformPixelXY = histc[9] * 3

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

                histn = hist(lbpVideos, \
                                tt - halfTemplate, tt + halfTemplate + 1, \
                                ii - halfTemplate, ii + halfTemplate + 1, \
                                jj - halfTemplate, jj + halfTemplate + 1, \
                                sizeXY, sizeXT, sizeYT)

                # chi-square dissimilarity measure
                wLBP[tw, iw, jw] =  np.sum(((histc - histn)**2) / (histc + histn))
                #wLBP[tw, iw, jw] = 0.0
                #w[tw, iw, jw] = 0.0

    # calc w_LBP
    hs = np.std(wLBP)
    wLBP = np.exp(-wLBP/hs)
    wLBP[halfWindowSize, halfWindowSize, halfWindowSize] = 0.0
    wLBP[halfWindowSize, halfWindowSize, halfWindowSize] = np.max(wLBP)
    wLBP = wLBP / np.sum(wLBP)

    # calc w_Intensity
    w[halfWindowSize, halfWindowSize, halfWindowSize] = 0.0
    w[halfWindowSize, halfWindowSize, halfWindowSize] = np.max(w)
    w = w / np.sum(w)

    m = w * wLBP
    m = m / np.sum(m)

    neighborhood = video[t - halfWindowSize: t + halfWindowSize + 1, \
                        i - halfWindowSize: i + halfWindowSize + 1, \
                        j - halfWindowSize: j + halfWindowSize + 1]

    aux = 0.0

    if nonUniformPixel >= 0.09:
        aux = np.sum(w*neighborhood)
    else:
        aux = np.sum(m*neighborhood)

    print 'Pixel (%3d, %3d) processed!!! ' % (i-delta, j-delta)
    #return np.sum(w*neighborhood),  np.sum(wLBP*neighborhood), np.sum(m*neighborhood), nonUniformPixel, nonUniformPixelXY
    return np.sum(w*neighborhood),  aux, np.sum(m*neighborhood), nonUniformPixel, nonUniformPixelXY

class FastParNLMeans3D:
    def __init__(self, h = 3, templateWindowSize = 7, searchWindowSize = 21, sigma = 1):
        self.h = h
        self.templateWindowSize = templateWindowSize
        self.searchWindowSize = searchWindowSize
        self.sigma = sigma


    def neighborhoodFeatures(self, video):
        nFrames = video.shape[0]
        nRows = video.shape[1]
        nCols = video.shape[2]

        halfWindowSize = self.searchWindowSize / 2
        halfTemplate = self.templateWindowSize / 2
        delta = halfWindowSize + halfTemplate

        avg = np.zeros(video.shape)
        grad = np.gradient(video)
        gradT = grad[0]
        gradX = grad[1]
        gradY = grad[2]
        avgGrad = np.zeros((video.shape[0], video.shape[1], video.shape[2], 3))

        for t in xrange(delta, nFrames - delta):
            for i in xrange(delta, nRows - delta):
                for j in xrange(delta, nCols - delta):
                    t1 = t - halfTemplate
                    t2 = t + halfTemplate + 1
                    i1 = i - halfTemplate
                    i2 = i + halfTemplate + 1
                    j1 = j - halfTemplate
                    j2 = j + halfTemplate + 1

                    pc = video[t1:t2, i1:i2, j1:j2]

                    gcT = gradT[t1:t2, i1:i2, j1:j2]
                    gcX = gradX[t1:t2, i1:i2, j1:j2]
                    gcY = gradY[t1:t2, i1:i2, j1:j2]

                    avg[t,i,j] = np.mean(pc)
                    avgGrad[t,i,j] = (np.mean(gcT), np.mean(gcX), np.mean(gcY))


        
        theta = np.array(
                    [ np.arccos(np.dot(avgGrad[t1,i1,j1], avgGrad[t2,i2,j2]) / (la.norm(avgGrad[t1,i1,j1]) * la.norm(avgGrad[t2,i2,j2])))
                    for t1, i1, j1 in self.coordinates for t2, i2, j2 in self.coordinates ]
        ).reshape(video.shape + video.shape)
        print 'ERRRRRRRRRRROU'
        return avg, avgGrad, theta

    # video -> numpy array (3D)
    def denoise(self, videoIn):
        halfWindowSize = self.searchWindowSize / 2
        halfTemplate = self.templateWindowSize / 2
        delta = halfWindowSize + halfTemplate

        shape = tuple(np.add(videoIn.shape, (0, 2*delta, 2*delta)))
        video = np.zeros(shape)
        for i in xrange(0, videoIn.shape[0]):
            video[i] = cv2.copyMakeBorder(videoIn[i], delta, delta, delta, delta, cv2.BORDER_REFLECT_101)

        out = video.copy()
        outLBP = video.copy()
        outM = video.copy()
        outNonUni = video.copy()
        outNonUniXY = video.copy()
        #out = np.ones(video.shape)

        nFrames = video.shape[0]
        nRows = video.shape[1]
        nCols = video.shape[2]

        aux = np.zeros((self.templateWindowSize, self.templateWindowSize, self.templateWindowSize))
        aux[halfTemplate, halfTemplate, halfTemplate] = 1
        gaussian = ndimage.filters.gaussian_filter(aux, self.sigma)

        ranges = [range(delta, nFrames - delta), range(delta, nRows - delta), range(delta, nCols - delta)]
        self.coordinates = list(itertools.product(*ranges))

        lbpTop = LBPTOP(LBP(8, uniform=True, rotation_invariant=True), \
                        LBP(8, uniform=True, rotation_invariant=True), \
                        LBP(8, uniform=True, rotation_invariant=True))
        lbpVideos = lbpTop.generateCodes(video)
        sizeXY = lbpTop.getMaxXY()
        sizeXT = lbpTop.getMaxXT()
        sizeYT = lbpTop.getMaxYT()

        # Fast Non Local Means
        avg, avgGrad = self.neighborhoodFeatures(video)

        ncpus = joblib.cpu_count()
        results = Parallel(n_jobs=ncpus,max_nbytes=2e9)(delayed(processPixel)(video, t, i, j, self.h, halfWindowSize, halfTemplate, gaussian, lbpVideos, sizeXY, sizeXT, sizeYT) for t,i,j in coordinates)

        for idx in xrange(0,len(results)):
            out[coordinates[idx][0], coordinates[idx][1], coordinates[idx][2]] = results[idx][0]
            outLBP[coordinates[idx][0], coordinates[idx][1], coordinates[idx][2]] = results[idx][1]
            outM[coordinates[idx][0], coordinates[idx][1], coordinates[idx][2]] = results[idx][2]
            outNonUni[coordinates[idx][0], coordinates[idx][1], coordinates[idx][2]] = results[idx][3]
            outNonUniXY[coordinates[idx][0], coordinates[idx][1], coordinates[idx][2]] = results[idx][4]

        return out[:, delta: -delta, delta: -delta], \
                outLBP[:, delta: -delta, delta: -delta], \
                outM[:, delta: -delta, delta: -delta],\
                outNonUni[:, delta: -delta, delta: -delta],\
                outNonUniXY[:, delta: -delta, delta: -delta]
