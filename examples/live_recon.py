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


# PHASE 1:
# preparation

# - geometry definition meta read
# - ASTRA projection data object persistent for 360 degrees
# - prepare CUDA BP algorithms

# PHASE 2:
# incoming data

# - a) flat fields
# - b) accept packets with projection data
#  - which angle
#  - projection data to upload to ASTRA
#  - flat field
#  - linearize

def add_projection(astra_id, flat, dark, proj, angle_idx):
    proj = (proj - dark) / (flat - dark)
    proj = -np.log(proj)
    astra.upload_projection(astra_id, angle_idx, proj, filt=True)

# PHASE 3:
# RECAST reconstuction server

# - as is
# - except:
#   - Update non-changing slices when enough new projection have arrived


def main():
    parser = argparse.ArgumentParser(
        description='Host a live reconstruction experiment.')
    parser.add_argument('name', metavar='NAME', type=str, default='Anonymous',
                        help="The name of the experiment")

    parser.add_argument('dir', metavar='DIR', type=str,
                        help="The directory of the FleX-Ray data")
    args = parser.parse_args()

    # prepare geometry
    flat = None
    dark = None
    proj = None

    # prepare volume
    N = 1024
    vol = np.zeros([1, N, N], dtype='float32')

    # ASTRA objects
    # vol_geom = flex.data.astra_vol_geom(
    #     meta['geometry'], vol.shape,  proj.shape[::2])
    # proj_geom = flex.data.astra_proj_geom(meta['geometry'], proj.shape[::2])
    # original_vectors = proj_geom['Vectors'].copy()
    # sin_id = astra.data3d.create(
    #     '-sino', proj_geom, np.ascontiguousarray(proj), persistent=True)
    # vol_id = astra.data3d.link('-vol', vol_geom, vol)


    def projection_callback(proj_data, angle):
        add_projection(sin_id, flat, dark, proj_data, angle)


    def callback(orientation, slice_id):
        print ('hi')
        new_vectors = slre.transforms.transform_vectors(original_vectors,
                                                        np.array(orientation), K)

        # get astra vecs
        proj_geom['Vectors'] = new_vectors
        astra.data3d.change_geometry(sin_id, proj_geom)
        astra.algorithm.run(slice_alg_id, 1)

        return [N, N], vol.ravel()

    # Note that the tomop server is really for subscribing to RECAST3D
    # How are we going to accept also projection data.. maybe we can just add
    # this to C++ too
    serv = tomop.server(args.name)
    serv.set_callback(callback)
    serv.serve()

if __name__ == "__main__":
    main()
