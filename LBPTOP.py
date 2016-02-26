# A different input/output interface for bob.ip.base.LBPTop

import numpy as np
import sys
import cv2

from bob.ip.base import * # Bob's implementation of LBP-TOP

class LBPTOP:
    def __init__(self, lbpXY, lbpXT, lbpYT):
        self.lbpTop = LBPTop(lbpXY, lbpXT, lbpYT)

        self.histSize = self.lbpTop.xy.max_label + \
                        self.lbpTop.xt.max_label + \
                        self.lbpTop.yt.max_label

    def getHistSize(self):
        return self.histSize

    def getMaxXY(self):
        return self.lbpTop.xy.max_label

    def getMaxXT(self):
        return self.lbpTop.xt.max_label

    def getMaxYT(self):
        return self.lbpTop.yt.max_label

    def generateCodes(self, cuboid):
        shape = tuple(np.add(cuboid.shape, (0, 2, 2)))

        newCuboid = np.zeros(shape)

        for i in xrange(0, cuboid.shape[0]):
            newCuboid[i] = cv2.copyMakeBorder(cuboid[i],1,1,1,1,cv2.BORDER_REFLECT_101)

        firstFrame = np.reshape(newCuboid[0], (1,newCuboid[0].shape[0],newCuboid[0].shape[1]))
        lastFrame = np.reshape(newCuboid[-1], (1,newCuboid[-1].shape[0],newCuboid[-1].shape[1]))
        newCuboid = np.concatenate((firstFrame,newCuboid,lastFrame))

        print newCuboid.shape
        print cuboid.shape

        xy = np.zeros(cuboid.shape, dtype=np.uint16)
        xt = np.zeros(cuboid.shape, dtype=np.uint16)
        yt = np.zeros(cuboid.shape, dtype=np.uint16)

        self.lbpTop.process(newCuboid, xy, xt, yt)

        return {'xy': xy, 'xt': xt, 'yt': yt}

    def describe(self, cuboid):
        codes = self.generateCodes(cuboid)

        hists = np.concatenate((np.histogram(codes['xy'], self.lbpTop.xy.max_label)[0], \
                                np.histogram(codes['xt'], self.lbpTop.xt.max_label)[0], \
                                np.histogram(codes['yt'], self.lbpTop.yt.max_label)[0]))

        return hists
