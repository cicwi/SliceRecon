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
    global flat
    global dark

    # PHASE 1:
    # preparation

    # - geometry definition meta read
    # - ASTRA projection data object persistent for 360 degrees
    # - prepare CUDA BP algorithms
    parser = argparse.ArgumentParser(
        description='Host a live reconstruction experiment.')
    parser.add_argument('name', metavar='NAME', type=str, default='Anonymous',
                        help="The name of the experiment")

    parser.add_argument('dir', metavar='DIR', type=str,
                        help="A directory of FleX - Ray data, used temporarily to extract"
                        "scanner settings(of already completed scan)")

    parser.add_argument('--bin', metavar='BIN', type=int, help="Binning factor",
                        default=1)

    parser.add_argument('--cpufilter', help="Filter on CPU",
                        action='store_true')

    args = parser.parse_args()

    meta = flex.data.read_log(args.dir, 'flexray')
    # find e.g. the dark field, and read shape out of that
    dark_file = flex.data._get_files_sorted_(args.dir, 'di')[0]
    image = flex.data._read_tiff_(dark_file)

    # get proj count
    proj_count = int(meta['geometry']['theta_count'])
    size = np.array(image.shape) // args.bin
    meta['geometry']['det_pixel'] *= args.bin

    print(proj_count, size)
    print('meta', meta)

    # prepare geometry
    flat = np.ones(size)
    dark = np.zeros(size)

    proj = np.zeros([size[0], proj_count,
                     size[1]],
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

    astra.algorithm.run(slice_alg_id)
    # PHASE 2:
    # incoming data
    global serv
    global count
    serv = tomop.server(args.name)
    count = 0

    # - a) flat fields
    # - b) accept packets with projection data
    #  - which angle
    #  - projection data to upload to ASTRA
    #  - flat field
    #  - linearize
    def projection_callback(proj_shape, proj_data, proj_id):
        global flat
        global dark
        global count
        global serv

        if proj_id > 0:
            proj_id = proj_id % proj_count

        if proj_id == -1:
            flat = np.reshape(proj_data, proj_shape)
        elif proj_id == -2:
            dark = np.reshape(proj_data, proj_shape)
        else:
            proj = (np.reshape(proj_data, proj_shape) - dark) / (flat - dark)
            proj = -np.log(proj)
            if args.cpufilter:
                proj = slre.util.prescale_and_filter(
                    original_vectors, proj, proj_id)
            astra.data3d.upload_projection(sin_id, proj_id, proj, filt=(not
                                                                        args.cpufilter))

        # TODO: START RECONSTRUCTING SLICES WHEN
        # TODO: AFTER ENOUGH NEW PROJECTIONS, REDO ALL CURRENT SLICES
        count = count + 1
        if count % 25 == 0:
            print('update', count)
            grsp = tomop.group_request_slices_packet(
                serv.scene_id(), 1)
            serv.send(grsp)

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
    serv.set_callback(callback)
    serv.set_projection_callback(projection_callback)
    serv.serve()

if __name__ == "__main__":
    main()
