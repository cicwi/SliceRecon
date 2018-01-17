# Depends on:
# - tomobox
# - ASTRA
# - TomoPackets

import numpy as np
import scipy
import astra
import flexbox as flex
import tomop
import slicerecon as slre
import slicerecon.transforms as tf

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
full_proj = flex.data.raw2astra(full_proj)

# bin
group = 2
proj = slre.util.bin(full_proj, group)
meta['geometry']['det_pixel'] *= group

print('Loaded:', name)

# ... proj.fix_dead_pix()

# TODO get something sensible from the data, allow overwriting
N = 1024
vol = np.zeros([1, N, N], dtype='float32')

vol_geom = flex.data.astra_vol_geom(
    meta['geometry'], vol.shape,  proj.shape[::2])
proj_geom = flex.data.astra_proj_geom(meta['geometry'], proj.shape[::2])
original_vectors = proj_geom['Vectors'].copy()
print('vol_geom', vol_geom)
print('vec', original_vectors[0])

slre.util.prescale_and_filter(original_vectors, proj)

sin_id = astra.data3d.create(
    '-sino', proj_geom, np.ascontiguousarray(proj), persistent=True)
vol_id = astra.data3d.link('-vol', vol_geom, vol)


# II. write conversion code from slice orientation definition to geometry
# rotation

K = vol_geom['option']['WindowMaxX']

# III. setup callback code + server
serv = tomop.server(name)

# a) do volume preview...
small_vol = np.zeros([128, 128, 128], dtype='float32')
small_vol_geom = flex.data.astra_vol_geom(
    meta['geometry'], small_vol.shape, proj.shape[::2])
small_vol_id = astra.data3d.link('-vol', small_vol_geom, small_vol)
small_projector_id = astra.create_projector(
    'cuda3d', proj_geom, small_vol_geom,  {'DensityWeighting': True})

cfg = astra.astra_dict('BP3D_CUDA')
cfg['ProjectorId'] = small_projector_id
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

projector_id = astra.create_projector(
    'cuda3d', proj_geom, vol_geom,  {'DensityWeighting': True})

cfg = astra.astra_dict('BP3D_CUDA')
cfg['ProjectorId'] = projector_id
cfg['ReconstructionDataId'] = vol_id
cfg['ProjectionDataId'] = sin_id

slice_alg_id = astra.algorithm.create(cfg)


def callback(orientation, slice_id):
    new_vectors = slre.transforms.transform_vectors(original_vectors,
                                                    np.array(orientation), K)

    # get astra vecs
    proj_geom['Vectors'] = new_vectors
    astra.data3d.change_geometry(sin_id, proj_geom)
    astra.algorithm.run(slice_alg_id, 1)

    #scipy.misc.imsave('slice.png', vol[0])

    return [N, N], vol.ravel()


serv.set_callback(callback)

print('Serving!')

serv.serve()

astra.data3d.delete(sin_id)
astra.data3d.delete(vol_id)
astra.data3d.delete(small_vol_id)
astra.algorithm.delete(slice_alg_id)
