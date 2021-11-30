import numpy as np
import math
from scipy.signal import convolve2d


def matlab_style_gauss2D(shape=(3, 3), sigma=0.5):
    m, n = [(ss - 1.) / 2. for ss in shape]
    y, x = np.ogrid[-m:m + 1, -n:n + 1]
    h = np.exp(-(x * x + y * y) / (2. * sigma * sigma))
    h[h < np.finfo(h.dtype).eps * h.max()] = 0
    sumh = h.sum()
    if sumh != 0:
        h /= sumh
    return h


def filter2(x, kernel, mode='same'):
    return convolve2d(x, np.rot90(kernel, 2), mode=mode)


def ssim(model, loader, k1=0.01, k2=0.03, win_size=11, L=255):
    ssim_map = 0
    for (inp, tar) in loader:
        pre = model(inp)
        im1 = inp.numpy()
        im2 = pre.numpy()
        M, N = im1.shape
        C1 = (k1 * L) ** 2
        C2 = (k2 * L) ** 2
        window = matlab_style_gauss2D(shape=(win_size, win_size), sigma=1.5)
        window = window / np.sum(np.sum(window))

        if im1.dtype == np.uint8:
            im1 = np.double(im1)
        if im2.dtype == np.uint8:
            im2 = np.double(im2)

        mu1 = filter2(im1, window, 'valid')
        mu2 = filter2(im2, window, 'valid')
        mu1_sq = mu1 * mu1
        mu2_sq = mu2 * mu2
        mu1_mu2 = mu1 * mu2
        sigma1_sq = filter2(im1 * im1, window, 'valid') - mu1_sq
        sigma2_sq = filter2(im2 * im2, window, 'valid') - mu2_sq
        sigmal2 = filter2(im1 * im2, window, 'valid') - mu1_mu2

        ssim_map += ((2 * mu1_mu2 + C1) * (2 * sigmal2 + C2)) / ((mu1_sq + mu2_sq + C1) * (sigma1_sq + sigma2_sq + C2))

    return np.mean(np.mean(ssim_map)) / len(loader.dataset)


def psnr(model, loader, scale):
    i = 1
    rmse = np.zeros(len(loader.dataset))
    for (inp, tar) in loader:
        target = model(inp).numpy()
        ref = tar.numpy()
        target_data = np.array(target)
        target_data = target_data[scale:-scale, scale:-scale]

        ref_data = np.array(ref)
        ref_data = ref_data[scale:-scale, scale:-scale]

        diff = ref_data - target_data
        diff = diff.flatten('C')
        rmse[i] = math.sqrt(np.mean(diff ** 2.))
        rmse[i] = 20 * math.log10(1.0 / rmse[i])
        i += 1
    return np.mean(rmse)
