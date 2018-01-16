from itertools import chain
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image
import odl
import os

# %% Data files
data_path = '/export/scratch2/kohr/data/rat_skull/'
offset_image_name = 'di0000.tif'
gain_image_name = 'io0000.tif'
num_projs = 1200
proj_image_names = ['scan_{:06}.tif'.format(i) for i in range(num_projs)]

# Check existence of all files upfront
for fname in chain([offset_image_name], [gain_image_name], proj_image_names):
    full_path = os.path.join(data_path, fname)
    if not os.path.exists(full_path):
        print('file {} does not exist'.format(full_path))

# %% Parameters from the scanner script

# Object
sample_size = 15  # (only x-y radius) [mm]

# Geometry
sdd = 281.000000  # [mm]
sod = 154.999512  # [mm]
first_angle = 0.0  # [deg]
last_angle = 360.0  # [deg]

# Detector
det_px_size = 0.149600  # (binned) [mm]
det_shape = (972, 768)

# Reconstruction (not necessarily needed)
voxel_size = 0.082519  # = px_size / magnification  [mm]
horiz_center = 481.283422  # [px]
vert_center = 387.700535  # [px]

# Region of interest (on detector, rest has no object info)
xmin = 292.000000  # [px]
xmax = 660.000000  # [px]
ymin = 292.000000  # [px]
ymax = 660.000000  # [px]
zmin = 98.000000  # [px]
zmax = 686.000000  # [px]

# %% Load data into a Numpy array

data_subset = np.s_[:]  # Define indexing object to get subsets
proj_image_names_subset = proj_image_names[data_subset]
num_projs_subset = len(proj_image_names_subset)
proj_data = np.empty((num_projs_subset,) + det_shape, dtype='float32')

for i in odl.util.ProgressRange('Reading projection data', num_projs_subset):
    fname = proj_image_names_subset[i]
    proj_image = Image.open(os.path.join(data_path, fname))
    proj_data[i] = np.rot90(proj_image, -1)


def neighbor(i, j):
    if i > 0:
        return i - 1, j
    elif j > 0:
        return i, j - 1
    else:
        return i + 1, j


# Fix dead pixels (very simple method)
for i in odl.util.ProgressRange('Fixing dead pixels', num_projs_subset):
    proj = proj_data[i]

    # We only expect a few dead pixels, so this won't take long
    dead_pixels = np.where(proj == 0)
    if np.size(dead_pixels) == 0:
        continue

    neighbors = [np.empty_like(dead_px_i) for dead_px_i in dead_pixels]
    for num, (i, j) in enumerate(zip(*dead_pixels)):
        inb, jnb = neighbor(i, j)
        neighbors[0][num] = inb
        neighbors[1][num] = jnb

    proj[dead_pixels] = proj[neighbors]

offset_image = np.rot90(
    Image.open(os.path.join(data_path, offset_image_name)), -1)

# plt.figure()
# plt.imshow(np.rot90(offset_image))
# plt.title('dark image')
# plt.show()

gain_image = np.rot90(Image.open(os.path.join(data_path, gain_image_name)), -1)

# plt.figure()
# plt.imshow(np.rot90(gain_image))
# plt.title('gain image')
# plt.show()

# Normalize data with gain & offset images, and take the negative log
for i in odl.util.ProgressRange('Applying log transform', num_projs_subset):
    proj_data[i] -= offset_image
    proj_data[i] /= gain_image - offset_image
    np.log(proj_data[i], out=proj_data[i])
    proj_data[i] *= -1

# Display a sample (indices only for full dataset)
# plt.figure()
# plt.subplot(221)
# plt.imshow(np.rot90(proj_data[0]))
# plt.title('0 degrees')
#
# plt.subplot(222)
# plt.imshow(np.rot90(proj_data[299]))
# plt.title('90 degrees')
#
# plt.subplot(223)
# plt.imshow(np.rot90(proj_data[599]))
# plt.title('180 degrees')
#
# plt.subplot(224)
# plt.imshow(np.rot90(proj_data[899]))
# plt.title('270 degrees')
#
# plt.tight_layout()
# plt.suptitle('Projection images')
# plt.show()

# %% Define ODL geometry

full_angle_partition = odl.uniform_partition(
    np.radians(first_angle), np.radians(last_angle), num_projs,
    nodes_on_bdry=True)

det_min_pt = -det_px_size * np.array(det_shape) / 2.0
det_max_pt = det_px_size * np.array(det_shape) / 2.0
det_partition = odl.uniform_partition(det_min_pt, det_max_pt, det_shape)

src_radius = sod
det_radius = sdd - sod
magnification = (src_radius + det_radius) / src_radius

# Shift between object center and rotation center
rot_xshift = (horiz_center - (xmax + xmin) / 2) * det_px_size / magnification
rot_zshift = (vert_center - (zmax + zmin) / 2) * det_px_size / magnification

full_geometry = odl.tomo.ConeFlatGeometry(
    full_angle_partition, det_partition, src_radius, det_radius,
    axis_shift=-np.array([rot_xshift, 0, rot_zshift]))

# %% ODL reco space

# Volume size in mm
vol_size = np.array([xmax - xmin, ymax - ymin, zmax - zmin])
vol_size *= det_px_size / magnification
vol_size *= 1.2  # safety margin

vol_shift = np.array([rot_xshift, 0, rot_zshift])
vol_min_pt = -vol_size / 2 + vol_shift
vol_max_pt = vol_size / 2 + vol_shift
vol_shape = (vol_size / det_px_size * magnification).astype(int)

#vol_size[2] = vol_size[2] / vol_shape[2]
#vol_shape[2] = 1
#z_shift = 2.0
#vol_min_pt[2] = -vol_size[2] / 2 + z_shift
#vol_max_pt[2] = vol_size[2] / 2 + z_shift

reco_space = odl.uniform_discr(vol_min_pt, vol_min_pt + vol_size, vol_shape,
                               dtype='float32')

# %% Ray transform operator

full_ray_trafo = odl.tomo.RayTransform(reco_space, full_geometry)
full_fbp_op = odl.tomo.fbp_op(full_ray_trafo, padding=False,
                              filter_type='Hann', frequency_scaling=0.95)
full_fbp_filter_op = odl.tomo.fbp_filter_op(
    full_ray_trafo, padding=False, filter_type='Hann', frequency_scaling=0.95)
filtered_data = full_fbp_filter_op(proj_data)
filtered_data = full_ray_trafo.range.element(filtered_data)
full_fbp_reco = full_ray_trafo.adjoint(filtered_data)

# fig = None
# for n in range(full_fbp_reco.shape[1]):
#     fig = full_fbp_reco.show(indices=[None, n, None], fig=fig, clim=[0, 0.5])
