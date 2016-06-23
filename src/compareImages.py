import numpy as np
from skimage.measure import structural_similarity

def psnr(im1, im2):
    aux = mse(im1, im2)
    aux = 20 * np.log10(np.max(im1) / np.sqrt(aux))
    return aux

def ssim(im1, im2):
    return structural_similarity(im1, im2, dynamic_range=im1.max() - im1.min())


def mse(im1, im2):
    mse = np.sum((im1 - im2)**2)
    return mse / (im1.shape[0] * im1.shape[1])
