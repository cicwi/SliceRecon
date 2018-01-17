import tomop
import flexbox as flex
import numpy as np
import argparse
import tqdm
import slicerecon as slre

def main():
    serv = tomop.publisher("tcp://localhost:5557")

    parser = argparse.ArgumentParser(
        description='Fake publish a FleX-Ray experiment.')
    parser.add_argument('dir', metavar='DIR', type=str,
                        help="Directory with FleX-Ray data")
    parser.add_argument('--bin', metavar='BIN', type=int, help="Binning factor",
                        default=1)
    args = parser.parse_args()

    full_proj, full_flat, full_dark, meta = flex.data.read_flexray(args.dir)
    full_proj = flex.data.raw2astra(full_proj)
    full_flat = flex.data.raw2astra(full_flat)
    full_dark = flex.data.raw2astra(full_dark)

    if args.bin != 1:
        full_proj = slre.util.bin(full_proj, args.bin)
        full_dark = slre.util.bin(full_dark, args.bin)
        full_flat = slre.util.bin(full_flat, args.bin)
        meta['geometry']['det_pixel'] *= args.bin

    shape = [full_proj.shape[0], full_proj.shape[2]]
    print('shape:', shape)

    flatpdp = tomop.projection_data_packet(
        -1,
        -1,
        list(np.zeros([3])),
        list(np.zeros([9])),
        shape,
        full_flat.mean(1).ravel())

    darkpdp = tomop.projection_data_packet(
        -1,
        -2,
        list(np.zeros([3])),
        list(np.zeros([9])),
        shape,
        full_dark.mean(1).ravel())

    serv.send(flatpdp)
    serv.send(darkpdp)

    for i in tqdm.tqdm(np.arange(full_proj.shape[1]), desc='projs'):
        pdp = tomop.projection_data_packet(
            -1,
            i,
            list(np.zeros([3])),
            list(np.zeros([9])),
            shape,
            full_proj[:, i, :].ravel())
        serv.send(pdp)

if __name__ == "__main__":
    main()
