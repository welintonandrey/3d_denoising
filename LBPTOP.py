# A different input/output interface for bob.ip.base.LBPTop

import numpy as np
import sys

from bob.ip.base import * # Bob's implementation of LBP-TOP

class LBPTOP:
    def __init__(self, lbpXY, lbpXT, lbpYT):
        #self.lbpTop = LBPTop(lbpXY, lbpXT, lbpYT)
        self.lbpTop = LBPTop(LBP(8), LBP(8), LBP(8))

        self.histSize = self.lbpTop.xy.max_label + \
                        self.lbpTop.xt.max_label + \
                        self.lbpTop.yt.max_label

    def getHistSize(self):
        return self.histSize

    def generateCodes(self, cuboid):
        shape = tuple(np.subtract(cuboid.shape, (2, 2, 2)))

        xy = np.zeros(shape, dtype=np.uint16)
        xt = np.zeros(shape, dtype=np.uint16)
        yt = np.zeros(shape, dtype=np.uint16)

        self.lbpTop.process(cuboid, xy, xt, yt)

        return {'xy': xy, 'xt': xt, 'yt': yt}

    def describe(self, cuboid):
        codes = self.generateCodes(cuboid)

        hists = np.concatenate((np.histogram(codes['xy'], self.lbpTop.xy.max_label)[0], \
                                np.histogram(codes['xt'], self.lbpTop.xt.max_label)[0], \
                                np.histogram(codes['yt'], self.lbpTop.yt.max_label)[0]))

        return hists
