# Geometry

There are two coordinate systems:

- ASTRA
- RECAST3D

# Data

## Projection data

The projection data is in stored as a numpy array. If `Y` is the vertical detector
coordinate, `X` the horizontal detector coordinate and `Z` the projection index,
then we have
```python
proj.shape == (Y, Z, X)
```
The reason for this choice is because it is the ASTRA convention.


### Projection packets

The 'ProjectionDataPacket' can be used for a number of reasons. The
`projection_id` decides the use:

- `-1`: flat field
- `-2`: dark field
- `>= 0`: projection at angle index `projection_id`

## Volume data

We follow the ASTRA convention, which is
```python
vol.shape == (Z, X, Y)
```

## Orientations
An orientation vector is a numpy array with
```python
orientation.shape == (9)
axis_1, axis_2, base = np.split(orientation, 3)
```
so it contains:
- base ('bottom left')
- tilt1
- tilt2

## Vectors geometry

For cone beam geometries, we have a list:
```python
vectors.shape = (COUNT, 12)
```
where the 4 * 3 = 12 components contain respectively:
- ray direction
- detector center
- tilt1
- tilt2
