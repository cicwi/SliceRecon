# This script should:
# - Accept connection from a 'projection provider'
# - Connect to RECAST3D
#
# To simplify:
# - Ignore 3d preview for now

import numpy as np
import tomop
import slicerecon as slre
import argparse
import astra
import flexbox as flex
import threading


def main():
    # PHASE 1:
    # preparation
    
    # - geometry definition meta read
    # - ASTRA projection data object persistent for 360 degrees
    # - prepare CUDA BP algorithms
    parser = argparse.ArgumentParser(
        description='Host a live reconstruction experiment.')
    parser.add_argument('name', metavar='NAME', type=str, default='Anonymous',
                        help="The name of the experiment")

    parser.add_argument('det_x', metavar='DET_X', type=int,
                        help="width of detector")

    parser.add_argument('det_y', metavar='DET_Y', type=int,
                        help="width of detector")

    parser.add_argument('dir', metavar='DIR', type=str,
                        help="A directory of FleX - Ray data, used temporarily to extract"
                        "scanner settings(of already completed scan)")
    args = parser.parse_args()

    meta = flex.data.read_log(args.dir, 'flexray')

    print('meta', meta)

    # prepare geometry
    flat = None
    dark = None

    proj = np.zeros([args.det_y, int(meta['geometry']['theta_count']),
                     args.det_x],
                    dtype=np.float32)

    # prepare volume
    N = 1024
    vol = np.zeros([1, N, N], dtype='float32')

    # ASTRA objects
    vol_geom = flex.data.astra_vol_geom(
        meta['geometry'], vol.shape,  proj.shape[::2])
    proj_geom = flex.data.astra_proj_geom(meta['geometry'], proj.shape[::2])
    original_vectors = proj_geom['Vectors'].copy()
    sin_id = astra.data3d.create(
        '-sino', proj_geom, np.ascontiguousarray(proj), persistent=True)
    vol_id = astra.data3d.link('-vol', vol_geom, vol)
    K = vol_geom['option']['WindowMaxX']

    projector_id = astra.create_projector(
        'cuda3d', proj_geom, vol_geom,  {'DensityWeighting': True})

    cfg = astra.astra_dict('BP3D_CUDA')
    cfg['ProjectorId'] = projector_id
    cfg['ReconstructionDataId'] = vol_id
    cfg['ProjectionDataId'] = sin_id

    slice_alg_id = astra.algorithm.create(cfg)

    # PHASE 2:
    # incoming data
    
    # - a) flat fields
    # - b) accept packets with projection data
    #  - which angle
    #  - projection data to upload to ASTRA
    #  - flat field
    #  - linearize
    def projection_callback(proj_shape, proj_data, proj_id):
        global dark
        global flat
        # TODO: START RECONSTRUCTING SLICES WHEN
        # TODO: AFTER ENOUGH NEW PROJECTIONS, REDO ALL CURRENT SLICES
        if proj_id == -1:
            flat = np.reshape(proj_data, proj_shape)
        elif proj_id == -2:
            dark = np.reshape(proj_data, proj_shape)
        else:
            if dark and flat:
                proj = (proj_data - dark) / (flat - dark)
                proj = -np.log(proj)
            astra.upload_projection(sin_id, proj_id, proj, filt=True)


    # PHASE 3:
    # RECAST reconstuction server
    
    # - as is
    # - except:
    #   - Update non-changing slices when enough new projection have arrived
    def callback(orientation, slice_id):
        new_vectors = slre.transforms.transform_vectors(original_vectors,
                                                        np.array(orientation), K)

        # get astra vecs
        proj_geom['Vectors'] = new_vectors
        astra.data3d.change_geometry(sin_id, proj_geom)
        astra.algorithm.run(slice_alg_id, 1)

        return [N, N], vol.ravel()

    # Start server, both projection and recast
    serv = tomop.server(args.name)
    serv.set_callback(callback)
    serv.set_projection_callback(projection_callback)
    serv.serve()

if __name__ == "__main__":
    main()
