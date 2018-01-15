# Depends on:
# - tomobox
# - ASTRA
# - TomoPackets

import numpy as np
import scipy
import astra
import flexbox as flex
import tomop
import transforms3d as tf

import sys

# I. convert from FlexRay log to ASTRA geometry
# a. read data

if len(sys.argv) == 1:
    print("USAGE: python", sys.argv[0], "<path to data>")
    exit(-1)

path = sys.argv[1]

full_proj, full_flat, full_dark, meta = flex.data.read_flexray(path)
name = meta['description']['comments']

print('Fielding projections')
full_proj = (full_proj - full_dark) / (full_flat.mean(0) - full_dark)
full_proj = -np.log(full_proj)

print('Binning and filtering')
group = 1
proj = np.zeros((full_proj.shape[0], full_proj.shape[1] // group,
                 full_proj.shape[2] // group))


def bin_image(image, binning):
    tmp = image.reshape((image.shape[0] // binning, binning,
                         image.shape[1] // binning, binning))
    return np.sum(tmp, axis=(1, 3))

for i in np.arange(0, full_proj.shape[0]):
    p = bin_image(full_proj[i, :, :], group)
    proj[i, :, :] = p

# we have binned, so detector pixels are bigger
meta['geometry']['det_pixel'] *= group

print('Loaded:', name)

# b. prepare, fix and filter
proj = flex.data.raw2astra(proj)

# ... proj.fix_dead_pix()

# TODO get something sensible from the data, allow overwriting
N = 1024
vol = np.zeros([1, N, N], dtype='float32')

vol_geom = flex.data.astra_vol_geom(
    meta['geometry'], vol.shape,  proj.shape[::2])
print('vol_geom', vol_geom)
proj_geom = flex.data.astra_proj_geom(meta['geometry'], proj.shape[::2])
original_vectors = proj_geom['Vectors'].copy()

for i in np.arange(0, proj.shape[1]):
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
        p[row, :] = scipy.ifft(f)
    proj[:, i, :] = p

sin_id = astra.data3d.create(
    '-sino', proj_geom, np.ascontiguousarray(proj))#, persistent=True)
vol_id = astra.data3d.link('-vol', vol_geom, vol)


# II. write conversion code from slice orientation definition to geometry
# rotation


def normalize(x):
    return x / np.linalg.norm(x)


# Given x, y, yields an rotation matrix R such that Rx = ||x|| * (y / ||y||)
def rotation_onto(x, y):
    z = normalize(x)
    w = normalize(y)
    axis = np.cross(z, w)
    print(z, w, axis)
    if not axis.any():
        return np.identity(3)
    alpha = np.dot(z, w)
    angle = np.arccos(alpha)
    rot = tf.axangles.axangle2mat(axis, angle)
    return rot


# Given base, axis_1, axis_2, yields an affine transformation matrix that
# places origin at  (base + 0.5 * (axis_1 + axis_2))

# The (physical) space

K = vol_geom['option']['WindowMaxX']
print('K', K)


def slice_transform(base, axis_1, axis_2):
    # maybe first transform base, axis1 and axis2 to astra coord system...
    base = base * K
    axis_1 = axis_1 * K
    axis_2 = axis_2 * K
    delta = base + 0.5 * (axis_1 + axis_2)
    print("delta", delta)

    rot = rotation_onto(axis_1, np.array([2 * K, 0.0, 0.0]))
    rot = np.dot(rotation_onto(np.dot(rot, axis_2), np.array([0.0, 2 * K, 0.0])),
                 rot)
    scale1 = np.array([1.0, 1.0, 1.0]) + \
        (2 * K / np.linalg.norm(axis_1) - 1.0) * np.array([1.0, 0.0, 0.0])
    scale2 = np.array([1.0, 1.0, 1.0]) + \
        (2 * K / np.linalg.norm(axis_2) - 1.0) * np.array([0.0, 1.0, 0.0])
    scale = scale1 * scale2
    print("scale", scale)
    return -delta, rot, scale


# III. setup callback code + server
serv = tomop.server(name)

# a) do volume preview...
small_vol = np.zeros([128, 128, 128], dtype='float32')
small_vol_geom = flex.data.astra_vol_geom(
    meta['geometry'], small_vol.shape, proj.shape[::2])
print('small_vol_geom', small_vol_geom)
small_vol_id = astra.data3d.link('-vol', small_vol_geom, small_vol)

cfg = astra.astra_dict('FDK_CUDA')
cfg['ReconstructionDataId'] = small_vol_id
cfg['ProjectionDataId'] = sin_id

alg_id = astra.algorithm.create(cfg)
astra.algorithm.run(alg_id, 1)
astra.algorithm.delete(alg_id)

# TODO
vdp = tomop.volume_data_packet(
    serv.scene_id(),
    list(small_vol.shape),
    small_vol.ravel())

serv.send(vdp)

cfg = astra.astra_dict('BP3D_CUDA')
cfg['option'] = {}
cfg['option']['DensityWeighting'] = True
cfg['ReconstructionDataId'] = vol_id
cfg['ProjectionDataId'] = sin_id

slice_alg_id = astra.algorithm.create(cfg)


def callback(orientation, slice_id):
    axis_1, axis_2, base = np.split(np.array(orientation), 3)
    delta, rot, scale = slice_transform(base, axis_1, axis_2)
    new_vectors = np.zeros(original_vectors.shape)
    for i in np.arange(0, new_vectors.shape[0]):
        # ray direction, detector center, tilt1, tilt2
        s, d, t1, t2 = np.split(original_vectors[i], 4)
        new_vectors[i, 0:3] = np.dot(rot, s + delta)
        new_vectors[i, 3:6] = np.dot(rot, d + delta)
        new_vectors[i, 6:9] = np.dot(rot, t1)
        new_vectors[i, 9:] = np.dot(rot, t2)

    # get astra vecs
    proj_geom['Vectors'] = new_vectors
    astra.data3d.change_geometry(sin_id, proj_geom)
    astra.algorithm.run(slice_alg_id, 1)

    scipy.misc.imsave('slice.png', vol[0])

    return [N, N], vol.ravel()


serv.set_callback(callback)
serv.serve()

astra.data3d.delete(sin_id)
astra.data3d.delete(vol_id)
astra.data3d.delete(small_vol_id)
astra.algorithm.delete(slice_alg_id)
