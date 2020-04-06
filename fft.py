import cv2
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm
import numpy as np
import math
from scipy import sparse
import os
import argparse


def dft_slow(x):
    """Compute the discrete Fourier Transform of the 1D array x"""
    x = x.copy()
    x = np.asarray(x, dtype=complex)
    N = x.shape[0]
    n = np.arange(N)
    k = n.reshape((N, 1))
    M = np.exp(-2j * np.pi * k * n / N)
    return np.dot(M, x)


def dft2_slow(x):
    x = x.copy()
    x_trans = x.transpose()
    x_col_transformed = np.asarray(x_trans, dtype=complex)
    for n, col in enumerate(x_trans):
        x_col_transformed[n] = dft_slow(col)
    x_col_transformed = x_col_transformed.transpose()
    x_transformed = np.asarray(x, dtype=complex)
    for m, row in enumerate(x_col_transformed):
        x_transformed[m] = dft_slow(row)
    return x_transformed


def inverse_dft_slow(x):
    """Compute the discrete Fourier Transform of the 1D array x"""
    x = np.asarray(x, dtype=complex)
    N = x.shape[0]
    n = np.arange(N)
    k = n.reshape((N, 1))
    M = (1 / N) * np.exp(2j * np.pi * k * n / N)
    return np.dot(M, x)


def inverse_dft2_slow(x):
    x_trans = x.transpose()
    x_col_transformed = np.asarray(x_trans, dtype=complex)
    for n, col in enumerate(x_trans):
        x_col_transformed[n] = inverse_dft_slow(col)
    x_col_transformed = x_col_transformed.transpose()
    x_transformed = np.asarray(x, dtype=complex)
    for m, row in enumerate(x_col_transformed):
        x_transformed[m] = inverse_dft_slow(row)
    return x_transformed


def dft2_fast(x):
    x = x.copy()
    x_trans = x.transpose()
    x_col_transformed = np.asarray(x_trans, dtype=complex)
    for n, col in enumerate(x_trans):
        x_col_transformed[n] = fft(col)
    x_col_transformed = x_col_transformed.transpose()
    x_transformed = np.asarray(x, dtype=complex)
    for m, row in enumerate(x_col_transformed):
        x_transformed[m] = fft(row)
    return x_transformed


def inverse_dft2_fast(x):
    x_trans = x.transpose()
    x_col_transformed = np.asarray(x_trans, dtype=complex)
    for n, col in enumerate(x_trans):
        x_col_transformed[n] = inverse_fft(col, True)
    x_col_transformed = x_col_transformed.transpose()
    x_transformed = np.asarray(x, dtype=complex)
    for m, row in enumerate(x_col_transformed):
        x_transformed[m] = inverse_fft(row, True)
    return x_transformed


def fft(x):
    """A recursive implementation of the 1D Cooley-Tukey FFT"""
    x = np.asarray(x, dtype=complex)
    n = x.shape[0]

    if n % 2 > 0:
        raise ValueError("size of x must be a power of 2")
    elif n <= 32:  # this cutoff should be optimized
        return dft_slow(x)
    else:
        x_even = fft(x[::2])
        x_odd = fft(x[1::2])
        factor = np.exp(-2j * np.pi * np.arange(n) / n)
        x_even = np.concatenate([x_even, x_even])
        x_odd = np.concatenate([x_odd, x_odd])
        return x_even + factor * x_odd


def inverse_fft(x, norm=True):
    """A recursive implementation of the 1D Cooley-Tukey IFFT"""
    x = np.asarray(x, dtype=complex)
    n = x.shape[0]

    split_threshold = 32

    if n % 2 > 0:
        raise ValueError("size of x must be a power of 2")
    elif n <= split_threshold:  # this cutoff should be optimized
        return inverse_dft_slow(x) * n
    else:
        x_even = inverse_fft(x[::2], False)
        x_odd = inverse_fft(x[1::2], False)
        factor = np.exp(2j * np.pi * np.arange(n) / n)
        x_even = np.concatenate([x_even, x_even])
        x_odd = np.concatenate([x_odd, x_odd])
        x = x_even + factor * x_odd
        if norm:
            x = (1 / n) * x
        return x


def fast_mode(img):
    transformed = dft2_fast(img)

    plt.figure(figsize=(15, 5))
    plt.subplot(121), plt.imshow(img, cmap="gray")
    plt.title("Original"), plt.xticks([]), plt.yticks([])
    plt.subplot(122), plt.imshow(np.abs(transformed), norm=LogNorm(vmin=5))
    plt.colorbar()
    plt.title("Log scaled fourier transform"), plt.xticks([]), plt.yticks([])
    plt.suptitle("Our result for FFT Transform", fontsize=22)
    plt.show()


def denoising_mode(img):
    transformed_original = dft2_fast(img)
    thresh_factor = 0.1
    transformed = transformed_original.copy()
    r, c = transformed.shape
    transformed[int(r * thresh_factor):int(r * (1 - thresh_factor))] = 0
    transformed[:, int(c * thresh_factor):int(c * (1 - thresh_factor))] = 0
    num_zeros = r*c - np.count_nonzero(transformed)
    back = inverse_dft2_fast(transformed).real
    plt.figure(figsize=(15, 5))
    plt.subplot(121), plt.imshow(img, cmap="gray")
    plt.title("Original"), plt.xticks([]), plt.yticks([])
    plt.subplot(122), plt.imshow(back, cmap="gray")
    plt.title("Denoised"), plt.xticks([]), plt.yticks([])
    plt.suptitle("Threshold: {}%".format(100 * thresh_factor), fontsize=22)
    plt.show()
    print('Number of zeros: ', num_zeros)
    print('Overall size: ', transformed_original.size)
    print('Percentage of zeros: ', (num_zeros / transformed_original.size) * 100)


def compression_mode(img):
    transformed_original = dft2_fast(img)
    data_csr = sparse.csr_matrix(transformed_original)
    sparse.save_npz("original.npz", data_csr)
    size = os.path.getsize("original.npz")
    plt.figure(figsize=(15, 5))
    plt.subplot(231), plt.imshow(img, cmap="gray")
    plt.title("Original, size={} bytes".format(size)), plt.xticks([]), plt.yticks([])
    compression_factors = [30, 60, 80, 90, 95]
    index_count = 2
    for factor in compression_factors:
        transformed = transformed_original.copy()
        thresh = np.percentile(abs(transformed), factor)
        transformed[abs(transformed) < thresh] = 0
        data_csr = sparse.csr_matrix(transformed)
        file_name = "@{}%.npz".format(factor)
        sparse.save_npz(file_name, data_csr)
        size = os.path.getsize(file_name)
        back = inverse_dft2_fast(transformed).real
        plt.subplot(2, 3, index_count), plt.imshow(back, cmap="gray")
        plt.title("@ {}%, size={} bytes".format(factor, size)), plt.xticks([]), plt.yticks([])
        index_count = index_count + 1
    plt.suptitle("Compression Levels", fontsize=22)
    plt.show()


def main():
    parser = argparse.ArgumentParser(description='Compute Fourier Transforms')
    parser.add_argument('-m', type=int, default=1, action='store', help='mode to be selected')
    parser.add_argument('-i', type=str, default='./moonlanding.png', action='store', help='image filename')

    args = parser.parse_args()
    img = args.i
    mode = args.m

    img = cv2.imread(img, cv2.IMREAD_GRAYSCALE)
    width = len(img[0])
    height = len(img)
    width = width if width == 2 ** (int(np.log2(width))) else 2 ** (int(np.log2(width)) + 1)
    height = height if height == 2 ** (int(np.log2(height))) else 2 ** int((np.log2(height)) + 1)
    img = cv2.resize(img, (width, height))

    if mode == 1:
        fast_mode(img)
    elif mode == 2:
        denoising_mode(img)
    elif mode == 3:
        compression_mode(img)
    return


if __name__ == "__main__":
    main()
