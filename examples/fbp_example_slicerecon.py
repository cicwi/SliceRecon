"""Single-slice FBP in 3D cone beam geometry for tomopackets.

This example computes the FBP in a single slice by pre-computing the
filtered data and then back-projecting to an arbitrary slice.
The slice is defined by a slice visualizer using a set of matrix entries
for the transformation from slice coordinates to world coordinates in
the following way::

    [x]     [a  d  g]     [u]
    [y]  =  [b  e  h]  *  [v]
    [z]     [c  f  i]     [1]
    [1]     [0  0  1]

The ``set_callback`` method of a ``tomop.server`` expects a callback as
prototyped below.
"""
import os

os.environ['OMP_NUM_THREADS'] = '2'

import numpy as np
import odl
import time
import astra
import math

import sys
tomop_path = '/export/scratch1/buurlage/code/tomography/tomopackets/python/'
if tomop_path not in sys.path:
    sys.path.append(tomop_path)
import tomop

DEBUG = True

#astra.log.setOutputScreen(astra.log.STDERR, astra.log.DEBUG)

serv = tomop.server('Slice FBP', 'tcp://mabi.ci.cwi.nl:5555', 'tcp://mabi.ci.cwi.nl:5556')

SCENE_ID = serv.scene_id()
COUNT = 1
PARTS = 1
N = 512
M = min(N, 1024)
DATA = True
FILTER = True

del serv

# %% Variables defining the problem geometry

# Volume
vol_min_pt = np.array([-N/2, -N/2, -M/2], dtype=float)
vol_extent = [N, N, M]
vol_max_pt = vol_min_pt + vol_extent
vol_shape = (N, N, M)
vol_half_extent = np.array(vol_extent, dtype=float) / 2

# Projection angles
num_angles = N
min_angle = 0
max_angle = 2 * np.pi
angles = np.linspace(min_angle, max_angle, num_angles, endpoint=False)
angle_partition = odl.nonuniform_partition(angles)

# Partition along the angles
num_parts = PARTS
num_angles_per_part = num_angles // num_parts
slices = [slice(i * num_angles_per_part, (i + 1) * num_angles_per_part)
          for i in range(num_parts - 1)]
slices.append(slice((num_parts - 1) * num_angles_per_part, None))
angle_parts = [angles[slc] for slc in slices]

# Detector
det_min_pt = np.array([-N/2, -M/2], dtype=float)
det_extent = [N, M]
det_max_pt = det_min_pt + det_extent
det_shape = (N, M)
detector_partition = odl.uniform_partition(det_min_pt, det_max_pt, det_shape)

# Further geometry parameters
src_radius = 10*N
det_radius = 0
axis = [0, 0, 1]

# Constructor for geometries and arguments by keyword
if False:
    geometry_type = odl.tomo.CircularConeFlatGeometry
    geometry_kwargs_base = {
        'apart': angle_partition,
        'dpart': detector_partition,
        'src_radius': src_radius,
        'det_radius': det_radius
        }
else:
    geometry_type = odl.tomo.Parallel3dAxisGeometry
    geometry_kwargs_base = {
        'apart': angle_partition,
        'dpart': detector_partition,
        }
geometry_kwargs_full = geometry_kwargs_base.copy()
geometry_kwargs_full['axis'] = axis

# Filter parameters
padding = False
filter_type = 'Shepp-Logan'
relative_freq_cutoff = 0.8

# Parameters for the slice (given in slice coordinates)
slice_min_pt = np.array([-N/2, -N/2])
slice_extent = [N, N]
slice_max_pt = slice_min_pt + slice_extent
slice_shape = (N, N)
min_val = 0.0
max_val = 1.0


def geometry_part(geom_type, geom_kwargs, slc):
    apart = geom_kwargs['apart']
    angles = apart.grid.coord_vectors[0]
    angles_sub = angles[slc]
    geom_kwargs_sub = geom_kwargs.copy()
    geom_kwargs_sub['apart'] = odl.nonuniform_partition(angles_sub)
    return geom_type(**geom_kwargs_sub)


def geometry_part_frommatrix(geom_type, geom_kwargs, matrix, slc):
    apart = geom_kwargs['apart']
    angles = apart.grid.coord_vectors[0]
    angles_sub = angles[slc]
    geom_kwargs_sub = geom_kwargs.copy()
    geom_kwargs_sub['apart'] = odl.nonuniform_partition(angles_sub)
    geom_kwargs_sub['init_matrix'] = matrix
    return geom_type.frommatrix(**geom_kwargs_sub)


# %% Define the full problem


# Full reconstruction space (volume) and projection geometry
reco_space_full = odl.uniform_discr(vol_min_pt, vol_max_pt, vol_shape,
                                    dtype='float32')
geometry_full = geometry_type(**geometry_kwargs_full)


if DATA:
    print("Generating phantom")
    # Create a discrete Shepp-Logan phantom (modified version)
    phantom = odl.phantom.shepp_logan(reco_space_full, modified=True)
    print("Generated phantom")

for i in range(COUNT):
  INDEX = i
  if i == COUNT - 1:
    break
  r = os.fork()
  if r == 0:
    break

my_id = INDEX

LOGFILE = open("/tmp/slicer.log.%s" % (INDEX,), "w")

astra.astra.set_gpu_index(INDEX)


def slice_spec_to_rot_matrix(slice_spec):
    """Convert a slice specification vector to a 3x3 rotation matrix.

    Parameters
    ----------
    slice_spec : array-like, ``shape=(9,)``
        Real numbers ``a, b, c, d, e, f, g, h, i`` defining the
        transformation from slice coordinates to world coordinates.

    Returns
    -------
    matrix : `numpy.ndarray`, shape ``(3, 3)``
        Matrix that rotates the world system such that the slice vectors
        ``(a, b, c)`` and ``(d, e, f)`` are after normalization transformed
        to ``(1, 0, 0)`` and ``(0, 1, 0)``, respectively.

    Notes
    -----
    The transformation from normalized slice coordinates to normalized
    world coordinates is as follows:

        .. math::
            :nowrap:

            \\begin{equation*}

            \\begin{pmatrix}
              x \\\\
              y \\\\
              z \\\\
              1
            \end{pmatrix}
            %
            =
            %
            \\begin{pmatrix}
              a & d & g \\\\
              b & e & h \\\\
              c & f & i \\\\
              0 & 0 & 1
            \end{pmatrix}
            %
            \\begin{pmatrix}
              u \\\\
              v \\\\
              1
            \end{pmatrix}

            \end{equation*}

    Thus, the vectors :math:`U = (a, b, c)` and :math:`V = (d, e, f)`
    span the local slice coordinate system.
    """
    # TODO: udpate doc, out of sync now
    a, b, c, d, e, f, g, h, i = np.asarray(slice_spec, dtype=float)

    vec_u = np.array([a, b, c])
    vec_u_norm = np.linalg.norm(vec_u)
    if vec_u_norm == 0:
        raise ValueError('`[a, b, c]` vector is zero')
    else:
        vec_u /= vec_u_norm

    vec_v = np.array([d, e, f])
    vec_v_norm = np.linalg.norm(vec_v)
    if vec_v_norm == 0:
        raise ValueError('`[d, e, f]` vector is zero')
    else:
        vec_v /= vec_v_norm

    # Complete matrix to a rotation matrix
    normal = np.cross(vec_u, vec_v)
    return np.vstack([vec_v, normal, vec_u])


# Ray transform and filtering operator
ray_trafo_full = odl.tomo.RayTransform(reco_space_full, geometry_full,
                                       impl='astra_cuda')


# Create filtering operators for the individual parts
geometries_split = [geometry_part(geometry_type, geometry_kwargs_full, slc)
                    for slc in slices]
ray_trafos_split = [odl.tomo.RayTransform(reco_space_full, geom,
                                          impl='astra_cuda')
                    for geom in geometries_split]

# %% Create raw and filtered data

if DATA:

    # Create projection data by calling the ray transform on the phantom
    proj_data_split = ray_trafos_split[my_id](phantom).asarray()

    if FILTER:
        filter_ops_split = [
            odl.tomo.analytic.filtered_back_projection.fbp_filter_op(
                    ray_trafo, padding, filter_type,
                    frequency_scaling=relative_freq_cutoff)
            for ray_trafo in ray_trafos_split]

        # Filter the split data part by part
        #proj_data_filtered_split = [
        #    np.ascontiguousarray(filter_op(data).asarray())
        #    for filter_op, data in zip(filter_ops_split, proj_data_split)]
        proj_data_filtered_split = np.ascontiguousarray(filter_ops_split[my_id](proj_data_split).asarray())
    else:
        proj_data_filtered_split = proj_data_split
else:
    vg = astra.create_vol_geom(2,2)
    a = astra.data2d.create('-vol', vg)
    a = astra.data2d.create('-vol', vg)
    a = astra.data2d.create('-vol', vg)
    a = astra.data2d.create('-vol', vg)
    proj_data_filtered_split = ray_trafos_split[my_id].range.zero().asarray()

shape = proj_data_filtered_split.shape
print(shape)
proj_geom = astra.create_proj_geom('cone', 1, 1, shape[2], shape[1],
                                   list(range(shape[0])), 10*N, 0)
proj_id = astra.data3d.create('-proj3d', proj_geom, np.ascontiguousarray(np.rollaxis(proj_data_filtered_split, 2, 0)), persistent=True)
assert(proj_id == 5)

    
# %% Define callback that reconstructs the slice specified by the server
def callback_fbp(slice_spec, slice_id):
    """Reconstruct the slice given by ``slice_spec``.

    Parameters
    ----------
    slice_spec : `numpy.ndarray`, ``shape=(9,)``
        Real numbers ``a, b, c, d, e, f, g, h, i`` defining the
        transformation from normalized slice coordinates to
        normalized world coordinates. See Notes for details.

    Returns
    -------
    shape_array : `numpy_ndarray`, ``dtype='int32', shape=(2,)``
        Number of sent values per axis. Always equal to ``slice_shape``
        from global scope.
    values : `numpy.ndarray`, ``dtype='uint32', shape=(np.prod(shape_array),)``
        Flattened array of reconstructed values to send. The order of
        flattening is row-major (``'C'``).

    Notes
    -----
    The callback uses the following variables from global scope:

        - ``slice_shape`` : Number of points per axis in the slice that is
          to be reconstructed
        - ``vol_extent`` : Physical extent of the reconstruction volume,
          used to scale the normalized translation to a physical shift
        - ``geometry_type`` : Geometry class used in this problem
        - ``proj_data_filtered`` : Pre-filtered projection data
    """
    # TODO: use logging to log events
    if DEBUG:
        print('')
        print('--- New slice requested ---')
        print('')
    time_at_start = time.time()
    slice_spec = np.asarray(slice_spec, dtype=float)
    a, b, c, d, e, f, g, h, i = slice_spec

    # Just return empty stuff if shape is wrong, don't crash
    # TODO: maybe it's better to crash?
    try:
        slice_spec = slice_spec.reshape((9,))
    except ValueError as err:
        print('Malformed slice specification: expected shape (9,), got'
              'shape {}'.format(slice_spec.shape))
        return (np.array([0, 0], dtype='int32'), np.array([], dtype='float'))

    # Construct rotated & translated geometry as defined by slice_spec
    geom_kwargs = geometry_kwargs_base.copy()
    rot_world = slice_spec_to_rot_matrix(slice_spec)
    if DEBUG:
        print('world rotation to align slice with x-z plane:')
        print(rot_world)

    vec_u = np.array([a, b, c])
    vec_v = np.array([d, e, f])
    orig_norm = np.array([g, h, i])
    if DEBUG:
        print('U vector:', vec_u)
        print('V vector:', vec_v)
        print('origin:', orig_norm)

    # Scale by half extent since normalized sizes are between -1 and 1
    slc_pt1 = orig_norm * vol_half_extent
    slc_pt2 = (orig_norm + vec_u + vec_v) * vol_half_extent
    slc_mid_pt = (slc_pt1 + slc_pt2) / 2
    translation = reco_space_full.partition.mid_pt - slc_mid_pt
    translation_in_slc_coords = rot_world.dot(translation)
    if DEBUG:
        print('slice mid_pt:', slc_mid_pt)
        print('translation (world sys):', translation)
        print('translation (slice sys):', translation_in_slc_coords)

    init_matrix = np.hstack([rot_world, translation_in_slc_coords[:, None]])
    if DEBUG:
        print('slice geometry init_matrix:')
        print(init_matrix)
    geometry_slice = geometry_part_frommatrix(
        geometry_type, geom_kwargs, init_matrix, slices[my_id])

    # Construct slice reco space with size 1 in the z axis
    slc_pt1_rot = rot_world.dot(slc_pt1)
    slc_pt2_rot = rot_world.dot(slc_pt2)
    slc_min_pt_rot = np.minimum(slc_pt1_rot, slc_pt2_rot)
    slc_max_pt_rot = np.maximum(slc_pt1_rot, slc_pt2_rot)
    slc_spc_min_pt = slc_min_pt_rot.copy()
    slc_spc_min_pt[1] = -reco_space_full.cell_sides[1] / 2
    slc_spc_max_pt = slc_max_pt_rot.copy()
    slc_spc_max_pt[1] = reco_space_full.cell_sides[1] / 2
    slc_spc_shape = np.ones(3, dtype=int)
    slc_spc_shape[[0, 2]] = slice_shape
    if DEBUG:
        print('slice pt1 (slice sys):', slc_pt1_rot)
        print('slice pt2 (slice sys):', slc_pt2_rot)
        print('slice min_pt (slice sys):', slc_min_pt_rot)
        print('slice max_pt (slice sys):', slc_max_pt_rot)
        print('slice space min_pt:', slc_spc_min_pt)
        print('slice space max_pt:', slc_spc_max_pt)
        print('slice spcae shape:', slc_spc_shape)
    reco_space_slice = odl.uniform_discr(
        slc_spc_min_pt, slc_spc_max_pt, slc_spc_shape, dtype='float32',
        axis_labels=['$x^*$', '$y^*$', '$z^*$'])

    if DEBUG:
        print('slice space:')
        print(reco_space_slice)

    # Define ray trafo on the slice, using the transformed geometry
    # TODO: use specialized back-end when available
    ray_trafo_slice = odl.tomo.RayTransform(reco_space_slice, geometry_slice,
                                            impl='astra_cuda')

    time_after_setup = time.time()
    setup_time_ms = 1e3 * (time_after_setup - time_at_start)
    if DEBUG:
        print(INDEX, 'time for setup: {:7.3f} ms'.format(setup_time_ms))

    # Compute back-projection with this ray transform
    fbp_reco_slice = ray_trafo_slice.adjoint(proj_data_filtered_split)
    if DEBUG:
        # fbp_reco_slice.show()
        pass

    time_after_comp = time.time()
    comp_time_ms = 1e3 * (time_after_comp - time_after_setup)
    total_time_ms = 1e3 * (time_after_comp - time_at_start)
    if DEBUG:
        print(INDEX, 'time for computation: {:7.3f} ms'.format(comp_time_ms))
    #print("time_at_start", time_at_start, file=LOGFILE)
    #print("time_after_setup", time_after_setup, file=LOGFILE)
    #print("time_after_comp", time_after_comp, file=LOGFILE)
    print(INDEX, slice_id, N, COUNT, 'total time: {:7.3f} ms'.format(total_time_ms), file=LOGFILE)
    LOGFILE.flush()

    # Output must be a numpy.ndarray with 'uint32' data type, we first clip
    # to `[min_val, max_val]` and then rescale to [1, uint32_max - 1]
    reco_clipped = np.clip(np.asarray(fbp_reco_slice), min_val, max_val)

    #np.save('/tmp/slice%s.npy' % slice_id, reco_clipped)

    # Returning the shape as an 'int32' array and the flattened values
    return (np.array(slice_shape, dtype='int32').tolist(),
            reco_clipped.ravel())

# %% Connect to local slicevis server and register the callback

# Connect to server at localhost. The second argument would be the URI(?).
serv = tomop.server(SCENE_ID, COUNT, 'tcp://mabi.ci.cwi.nl:5555', 'tcp://mabi.ci.cwi.nl:5556')
#serv = tomop.server("FBP Slice", 'tcp://tethys.ci.cwi.nl:5555', 'tcp://tethys.ci.cwi.nl:5556')
print('Server started')

def rotation_matrix(axis, theta):
    """
    Return the rotation matrix associated with counterclockwise rotation about
    the given axis by theta radians.
    """
    axis = np.asarray(axis)
    axis = axis / math.sqrt(np.dot(axis, axis))
    a = math.cos(theta / 2.0)
    b, c, d = -axis * math.sin(theta / 2.0)
    aa, bb, cc, dd = a * a, b * b, c * c, d * d
    bc, ad, ac, ab, bd, cd = b * c, a * d, a * c, a * b, b * d, c * d
    return np.array([[aa + bb - cc - dd, 2 * (bc + ad), 2 * (bd - ac)],
                     [2 * (bc - ad), aa + cc - bb - dd, 2 * (cd + ab)],
                     [2 * (bd + ac), 2 * (cd - ab), aa + dd - bb - cc]])


if INDEX == 0:
    reco_space_preview = odl.uniform_discr(vol_min_pt, vol_max_pt, (64, 64, math.floor(64 * M / N)))
    reco_space_preview_swapped = odl.uniform_discr(vol_min_pt[::-1], vol_max_pt[::-1], (math.floor(64 * M / N), 64, 64))

    # Define a preview (coarse resolution volume) and quantize to uint32
    # TODO: replace with FBP
    preview = odl.phantom.shepp_logan(reco_space_preview, modified=True)
    preview_swapped = reco_space_preview_swapped.element(np.transpose(preview.asarray(), (2, 1, 0)))
    preview_quant = np.clip(np.asarray(preview_swapped), 0, 1)

    # Define volume packet and send it
    preview_packet = tomop.volume_data_packet(
        serv.scene_id(),
        np.array(preview_quant.shape[::-1], dtype='int32').tolist(),
        preview_quant.ravel())
    serv.send(preview_packet)
    print('Preview volume sent')

    for i in range(num_angles):
        center = np.array([0.0, 0.0, 0.0])
        source = [-10.0, 0.0, 0.0]
        detector_base = [3.0, -1.0, -1.0]
        detector_axes = [[0.0, 0.0, 2.0], [0.0, 2.0, 0.0]]
        rot = rotation_matrix([0, 0, 1], angles[i])
        rot_source = (np.dot(rot, source - center) + center).tolist()
        rot_a1 = np.dot(rot, detector_axes[0]).tolist()
        rot_a2 = np.dot(rot, detector_axes[1]).tolist()
        rot_base = (np.dot(rot, detector_base - center) + center).tolist()
        pdp = tomop.projection_data_packet(
                SCENE_ID, i, rot_source,
                rot_a1 + rot_a2 + rot_base, [1, 1],
                np.array([0.0], dtype='float32'))
        serv.send(pdp)


# Register the callback
# serv.set_callback(callback_null)
# print('NULL callback registered')
serv.set_callback(callback_fbp)
print('FBP callback registered')

# Do it
serv.serve()
