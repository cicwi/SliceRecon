import transforms3d as tf
import numpy as np


# Normalize a vector
def normalize(x):
    return x / np.linalg.norm(x)


# Given x, y, yields an rotation matrix R such that Rx = ||x|| * (y / ||y||)
def rotation_onto(x, y):
    z = normalize(x)
    w = normalize(y)
    axis = np.cross(z, w)
    if not axis.any():
        return np.identity(3)
    alpha = np.dot(z, w)
    angle = np.arccos(alpha)
    rot = tf.axangles.axangle2mat(axis, angle)
    return rot


# Transform a list of vectors
def slice_transform(base, axis_1, axis_2, K):
    # maybe first transform base, axis1 and axis2 to astra coord system...
    base = base * K
    axis_1 = axis_1 * K
    axis_2 = axis_2 * K
    delta = base + 0.5 * (axis_1 + axis_2)

    rot = rotation_onto(axis_1, np.array([2 * K, 0.0, 0.0]))
    rot = np.dot(rotation_onto(np.dot(rot, axis_2), np.array([0.0, 2 * K, 0.0])),
                 rot)
    scale1 = np.array([1.0, 1.0, 1.0]) + \
        (2 * K / np.linalg.norm(axis_1) - 1.0) * np.array([1.0, 0.0, 0.0])
    scale2 = np.array([1.0, 1.0, 1.0]) + \
        (2 * K / np.linalg.norm(axis_2) - 1.0) * np.array([0.0, 1.0, 0.0])
    scale = scale1 * scale2
    return -delta, rot, scale


# Transform a list of vectors
# TODO: add ASTRA convention to doc
def transform_vectors(original_vectors, orientation, K):
    axis_1, axis_2, base = np.split(orientation, 3)
    delta, rot, scale = slice_transform(base, axis_1, axis_2, K)
    new_vectors = np.zeros(original_vectors.shape)
    for i in np.arange(0, new_vectors.shape[0]):
        # ray direction, detector center, tilt1, tilt2
        s, d, t1, t2 = np.split(original_vectors[i], 4)
        new_vectors[i, 0:3] = scale * np.dot(rot, s + delta)
        new_vectors[i, 3:6] = scale * np.dot(rot, d + delta)
        new_vectors[i, 6:9] = scale * np.dot(rot, t1)
        new_vectors[i, 9:] = scale * np.dot(rot, t2)
    return new_vectors
