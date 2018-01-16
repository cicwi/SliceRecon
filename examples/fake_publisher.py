import tomop
import flexbox as flex
import numpy as np

serv = tomop.publisher("tcp://localhost:5557")

pdp = tomop.projection_data_packet(
    -1,
    0,
    list(np.zeros([3])),
    list(np.zeros([9])),
    list([1000, 1000]),
    np.ones([1000, 1000], dtype=np.float32).ravel())

serv.send(pdp)
