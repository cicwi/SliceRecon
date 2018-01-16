import numpy as np
import scipy
import tqdm


# Bin
def bin(full_proj, group):
    new_proj = np.zeros((full_proj.shape[0] // group, full_proj.shape[1],
                     full_proj.shape[2] // group), dtype='float32')

    def bin_image(image, binning):
        tmp = image.reshape((image.shape[0] // binning, binning,
                             image.shape[1] // binning, binning))
        return np.sum(tmp, axis=(1, 3))

    for i in tqdm.tqdm(np.arange(0, full_proj.shape[1]), desc='bin proj'):
        p = bin_image(full_proj[:, i, :], group)
        new_proj[:, i, :] = p

    return new_proj


# Filter
def prescale_and_filter(original_vectors, proj):
    for i in tqdm.tqdm(np.arange(0, proj.shape[1]), desc='filter proj'):
        p = proj[:, i, :]
        s, d, t1, t2 = np.split(original_vectors[i], 4)
        n = np.cross(t1, t2)
        rho = np.linalg.norm(s - np.dot(s - d, n) * n)
        xs, ys = np.meshgrid(np.arange(proj.shape[2]),
                             np.arange(proj.shape[0]))
        denums = np.linalg.norm((d - s) + np.outer(xs, t1) + np.outer(ys, t2))
        p = p * rho / denums

        # filter each row
        for row in np.arange(0, proj.shape[0]):
            values = p[row, :]
            f = scipy.fft(values)
            mid = (len(f) + 1) // 2
            f[:mid] = f[:mid] * np.arange(0, mid)
            f[mid:] = f[mid:] * np.arange(mid, 0, -1)
            p[row, :] = np.real(scipy.ifft(f))
        proj[:, i, :] = p
