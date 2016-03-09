import cv2
import sys
import numpy as np
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

def processPixel(video, t, i, j, h, halfWindowSize, halfTemplate, gaussian, lbpVideos, lbpVideosMSB, sizeXY, sizeXT, sizeYT, videoMSB):
    delta = halfWindowSize + halfTemplate
    ws = 2*halfWindowSize+1

    w_viORI = np.zeros((ws, ws, ws))
    w_viMSB = np.zeros((ws, ws, ws))
    w_texLBP = np.zeros((ws, ws, ws))
    w_texLBPMSB = np.zeros((ws, ws, ws))

    ta = t - halfTemplate
    tb = t + halfTemplate + 1
    ia = i - halfTemplate
    ib = i + halfTemplate + 1
    ja = j - halfTemplate
    jb = j + halfTemplate + 1

    pc = video[ta: tb, ia: ib, ja: jb]
    pcMSB = videoMSB[ta: tb, ia: ib, ja: jb]

    histc = hist(lbpVideos, ta, tb, ia, ib, ja, jb, sizeXY, sizeXT, sizeYT)
    histcMSB = hist(lbpVideosMSB, ta, tb, ia, ib, ja, jb, sizeXY, sizeXT, sizeYT)

    for tt in xrange(t - halfWindowSize, t + halfWindowSize + 1):
        for ii in xrange(i - halfWindowSize, i + halfWindowSize + 1):
            for jj in xrange(j - halfWindowSize, j + halfWindowSize + 1):
                t1 = tt - halfTemplate
                t2 = tt + halfTemplate + 1
                i1 = ii - halfTemplate
                i2 = ii + halfTemplate + 1
                j1 = jj - halfTemplate
                j2 = jj + halfTemplate + 1

                pn = video[t1: t2, i1: i2, j1: j2]
                pnMSB = videoMSB[t1: t2, i1: i2, j1: j2]

                tw = tt - t + halfWindowSize
                iw = ii - i + halfWindowSize
                jw = jj - j + halfWindowSize

                #Calc LBPHistogram ORIGINALvideo and MSBvideo
                histn = hist(lbpVideos, t1, t2, i1, i2, j1, j2, sizeXY, sizeXT, sizeYT)
                histnMSB = hist(lbpVideosMSB, t1, t2, i1, i2, j1, j2, sizeXY, sizeXT, sizeYT)

                # Calc chi-square dissimilarity measure
                w_texLBP[tw, iw, jw] =  np.sum(((histc - histn)**2) / (histc + histn))
                w_texLBPMSB[tw, iw, jw] =  np.sum(((histcMSB - histnMSB)**2) / (histcMSB + histnMSB))

                #Distances NLM ORIGINALvideo
                dist2 = np.sum(((pc-pn)**2)*gaussian)
                w_viORI[tw, iw, jw] = np.exp(-(dist2/(h*h)))

                #Distances NLM MSBvideo
                dist2 = np.sum(((pcMSB-pnMSB)**2)*gaussian)
                w_viMSB[tw, iw, jw] = np.exp(-(dist2/(h*h)))

    # Normalize weights matrix (video = Original, texture = None)
    w_viORI[halfWindowSize, halfWindowSize, halfWindowSize] = 0.0
    w_viORI[halfWindowSize, halfWindowSize, halfWindowSize] = np.max(w_viORI)
    w_viORI = w_viORI / np.sum(w_viORI)

    # Normalize weights matrix (video = MSB, texture = None)
    w_viMSB[halfWindowSize, halfWindowSize, halfWindowSize] = 0.0
    w_viMSB[halfWindowSize, halfWindowSize, halfWindowSize] = np.max(w_viMSB)
    w_viMSB = w_viMSB / np.sum(w_viMSB)

    # Normalize weights matrix (video = None, texture = LBP)
    # Aux (Not returned)
    hs = np.std(w_texLBP)
    w_texLBP = np.exp(-w_texLBP/hs)
    w_texLBP[halfWindowSize, halfWindowSize, halfWindowSize] = 0.0
    w_texLBP[halfWindowSize, halfWindowSize, halfWindowSize] = np.max(w_texLBP)
    w_texLBP = w_texLBP / np.sum(w_texLBP)

    # Normalize weights matrix (video = None, texture = LBP+MSB)
    # Aux (Not returned)
    hs = np.std(w_texLBPMSB)
    w_texLBPMSB = np.exp(-w_texLBPMSB/hs)
    w_texLBPMSB[halfWindowSize, halfWindowSize, halfWindowSize] = 0.0
    w_texLBPMSB[halfWindowSize, halfWindowSize, halfWindowSize] = np.max(w_texLBPMSB)
    w_texLBPMSB = w_texLBPMSB / np.sum(w_texLBPMSB)

    # Calc and Normalize weights matrix (video = Original, texture = LBP)
    w_viORI_texLBP = w_viORI * w_texLBP
    w_viORI_texLBP = w_viORI_texLBP / np.sum(w_viORI_texLBP)

    # Calc and Normalize weights matrix (video = MSB, texture = LBP)
    w_viMSB_texLPB = w_viMSB * w_texLBP
    w_viMSB_texLPB = w_viMSB_texLPB / np.sum(w_viMSB_texLPB)

    # Calc and Normalize weights matrix (video = MSB, texture = LBP+MSB)
    w_viMSB_texLPBMSB = w_viMSB * w_texLBPMSB
    w_viMSB_texLPBMSB = w_viMSB_texLPBMSB / np.sum(w_viMSB_texLPBMSB)

    # Calc and Normalize weights matrix (video = Original, texture = LBP+MSB)
    w_viORI_texLPBMSB = w_viORI * w_texLBPMSB
    w_viORI_texLPBMSB = w_viORI_texLPBMSB / np.sum(w_viORI_texLPBMSB)


    # Get Neighborhood
    neighborhood = video[t - halfWindowSize: t + halfWindowSize + 1, \
                        i - halfWindowSize: i + halfWindowSize + 1, \
                        j - halfWindowSize: j + halfWindowSize + 1]
    
    # Calc and Normalize weights matrix (video = Ori, texture = LBP Adaptive)
    nonUniformPixel = (histc[9] + histc[19] + histc[29])
    w_viORI_texLBPAdaptive = 0.0
    if nonUniformPixel >= 0.09:
        w_viORI_texLBPAdaptive = np.sum(w_viORI*neighborhood)
    else:
        w_viORI_texLBPAdaptive = np.sum(w_viORI_texLBP*neighborhood)

    # Calc and Normalize weights matrix (video = Ori, texture = LBP+MSB Adaptive)
    nonUniformPixel = (histcMSB[9] + histcMSB[19] + histcMSB[29])
    w_viORI_texLBPMSBAdaptive = 0.0
    if nonUniformPixel >= 0.09:
        w_viORI_texLBPMSBAdaptive = np.sum(w_viORI*neighborhood)
    else:
        w_viORI_texLBPMSBAdaptive = np.sum(w_viORI_texLPBMSB*neighborhood)

    #print 'Pixel (%3d, %3d) processed!!! ' % (i-delta, j-delta)
    totalPixel = (video.shape[1] - 2*delta) * (video.shape[2] - 2*delta)
    auxP = ((i-delta) * video.shape[2] + (j-delta))
    auxP = ((100 * auxP) / totalPixel)
    auxP = 0 if auxP<0 else 100 if i>100 else auxP
    sys.stdout.write('\rProcessando: %3d%% ' % auxP)
    sys.stdout.flush()

    #Return results
    return np.sum(w_viORI * neighborhood), \
            np.sum(w_viORI_texLBP * neighborhood), \
            np.sum(w_viORI_texLPBMSB * neighborhood), \
            w_viORI_texLBPAdaptive, \
            w_viORI_texLBPMSBAdaptive, \
            np.sum(w_viMSB * neighborhood), \
            np.sum(w_viMSB_texLPB * neighborhood), \
            np.sum(w_viMSB_texLPBMSB * neighborhood)

class ParNLMeans3D:
    def __init__(self, h = 3, templateWindowSize = 7, searchWindowSize = 21, sigma = 1, nMSB = 4):
        self.h = h
        self.templateWindowSize = templateWindowSize
        self.searchWindowSize = searchWindowSize
        self.sigma = sigma
        self.nMSB = nMSB

    # video -> numpy array (3D)
    def denoise(self, videoIn):
        halfWindowSize = self.searchWindowSize / 2
        halfTemplate = self.templateWindowSize / 2
        delta = halfWindowSize + halfTemplate

        shape = tuple(np.add(videoIn.shape, (0, 2*delta, 2*delta)))
        video = np.zeros(shape)
        for i in xrange(0, videoIn.shape[0]):
            video[i] = cv2.copyMakeBorder(videoIn[i], delta, delta, delta, delta, cv2.BORDER_REFLECT_101)

        out_viORI = video.copy()
        out_texLBP = video.copy()
        out_texLBPMSB = video.copy()
        out_viORI_texLBPAdaptive = video.copy()
        out_viORI_texLBPMSBAdaptive = video.copy()
        out_viMSB = video.copy()
        out_viMSB_texLBP = video.copy()
        out_viMSB_texLBPMSB = video.copy()

        nFrames = video.shape[0]
        nRows = video.shape[1]
        nCols = video.shape[2]

        aux = np.zeros((self.templateWindowSize, self.templateWindowSize, self.templateWindowSize))
        aux[halfTemplate, halfTemplate, halfTemplate] = 1
        gaussian = ndimage.filters.gaussian_filter(aux, self.sigma)

        ranges = [range(delta, nFrames - delta), range(delta, nRows - delta), range(delta, nCols - delta)]
        coordinates = list(itertools.product(*ranges))

        mask = 255 - (2**(8-self.nMSB) - 1)
        videoMSB = np.float64(np.uint8(video) & mask)

        lbpTop = LBPTOP(LBP(8, uniform=True, rotation_invariant=True), \
                        LBP(8, uniform=True, rotation_invariant=True), \
                        LBP(8, uniform=True, rotation_invariant=True))

        lbpVideos = lbpTop.generateCodes(video)
        lbpVideosMSB = lbpTop.generateCodes(videoMSB)

        sizeXY = lbpTop.getMaxXY()
        sizeXT = lbpTop.getMaxXT()
        sizeYT = lbpTop.getMaxYT()

        ncpus = joblib.cpu_count()
        results = Parallel(n_jobs=ncpus,max_nbytes=2e9)(delayed(processPixel)(video, t, i, j, self.h, halfWindowSize, halfTemplate, gaussian, lbpVideos, lbpVideosMSB, sizeXY, sizeXT, sizeYT, videoMSB) for t,i,j in coordinates)

        for idx in xrange(0,len(results)):
            out_viORI[coordinates[idx][0], coordinates[idx][1], coordinates[idx][2]] = results[idx][0]
            out_texLBP[coordinates[idx][0], coordinates[idx][1], coordinates[idx][2]] = results[idx][1]
            out_texLBPMSB[coordinates[idx][0], coordinates[idx][1], coordinates[idx][2]] = results[idx][2]
            out_viORI_texLBPAdaptive[coordinates[idx][0], coordinates[idx][1], coordinates[idx][2]] = results[idx][3]
            out_viORI_texLBPMSBAdaptive[coordinates[idx][0], coordinates[idx][1], coordinates[idx][2]] = results[idx][4]
            out_viMSB[coordinates[idx][0], coordinates[idx][1], coordinates[idx][2]] = results[idx][5]
            out_viMSB_texLBP[coordinates[idx][0], coordinates[idx][1], coordinates[idx][2]] = results[idx][6]
            out_viMSB_texLBPMSB[coordinates[idx][0], coordinates[idx][1], coordinates[idx][2]] = results[idx][7]

        return out_viORI[:, delta: -delta, delta: -delta], \
                out_texLBP[:, delta: -delta, delta: -delta], \
                out_texLBPMSB[:, delta: -delta, delta: -delta], \
                out_viORI_texLBPAdaptive[:, delta: -delta, delta: -delta],\
                out_viORI_texLBPMSBAdaptive[:, delta: -delta, delta: -delta],\
                out_viMSB[:, delta: -delta, delta: -delta],\
                out_viMSB_texLBP[:, delta: -delta, delta: -delta],\
                out_viMSB_texLBPMSB[:, delta: -delta, delta: -delta]
