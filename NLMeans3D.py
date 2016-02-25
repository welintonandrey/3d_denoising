import cv2
import sys
import numpy as np
from  scipy import ndimage

class NLMeans3D:
    def __init__(self, h = 3, templateWindowSize = 7, searchWindowSize = 21, sigma = 1):
        self.h = h
        self.templateWindowSize = templateWindowSize
        self.searchWindowSize = searchWindowSize
        self.sigma = sigma

    # video -> numpy array (3D)
    def denoise(self, video):
        out = video.copy()

        nFrames = video.shape[0]
        nRows = video.shape[1]
        nCols = video.shape[2]

        halfWindowSize = self.searchWindowSize / 2
        halfTemplate = self.templateWindowSize / 2
        delta = halfWindowSize + halfTemplate

        aux = np.zeros((self.templateWindowSize, self.templateWindowSize, self.templateWindowSize))
        aux[halfTemplate, halfTemplate, halfTemplate] = 1
        gaussian = ndimage.filters.gaussian_filter(aux, self.sigma)

        w = np.zeros((self.searchWindowSize, self.searchWindowSize, self.searchWindowSize))

        for t in xrange(delta, nFrames - delta):
            #print '[%5d] - Processing frame %d ... ' % (t, t),
            sys.stdout.flush()
            for i in xrange(delta, nRows - delta):
                for j in xrange(delta, nCols - delta):
                    print '(%d,%d,%d)' % (t,i,j)
                    pc = video[t - halfTemplate: t + halfTemplate + 1, \
                                i - halfTemplate: i + halfTemplate + 1, \
                                j - halfTemplate: j + halfTemplate + 1]
                    #mpc = np.mean(pc)

                    print 'Processing pixel (%3d, %3d) ... ' % (i, j),
                    for tt in xrange(t - halfWindowSize, t + halfWindowSize + 1):
                        for ii in xrange(i - halfWindowSize, i + halfWindowSize + 1):
                            for jj in xrange(j - halfWindowSize, j + halfWindowSize + 1):
                                pn = video[tt - halfTemplate: tt + halfTemplate + 1, \
                                            ii - halfTemplate: ii + halfTemplate + 1, \
                                            jj - halfTemplate: jj + halfTemplate + 1]

                                tw = tt - t + halfWindowSize
                                iw = ii - i + halfWindowSize
                                jw = jj - j + halfWindowSize

                                #w[tw, iw, jw] = 0.0
                                #mpn = np.mean(pn)
                                #ratio = mpc / mpn
                                #if ratio > 0.9 and ratio < 1.1:
                                #    dist = np.linalg.norm(pc-pn)
                                #    w[tw, iw, jw] = np.exp(-(dist*dist)/(self.h*self.h))

                                #print '(%d,%d,%d)' % (tw, iw, jw)
                                #dist = np.linalg.norm(pc-pn)
                                dist2 = np.sum(((pc-pn)**2)*gaussian)
                                #w[tw, iw, jw] = np.exp(-(dist*dist)/(self.h*self.h))
                                w[tw, iw, jw] = np.exp(-(dist2/(self.h*self.h)))

                    w[halfWindowSize, halfWindowSize, halfWindowSize] = 0.0
                    w[halfWindowSize, halfWindowSize, halfWindowSize] = np.max(w)

                    #print w
                    w = w / np.sum(w)
                    #print '------------------------------------'
                    #print w
                    #print 'sum = %lf' % (np.sum(w))
                    neighborhood = video[t - halfWindowSize: t + halfWindowSize + 1, \
                                        i - halfWindowSize: i + halfWindowSize + 1, \
                                        j - halfWindowSize: j + halfWindowSize + 1]
                    out[t,i,j] = np.sum(w*neighborhood)
                    print 'Done!'
            #print 'Done!'
        return out
