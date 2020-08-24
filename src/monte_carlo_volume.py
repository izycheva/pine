import numpy as np
import shape_utils

# ranges are the ranges from the invariant, given as a dictionary: {'x': (-1.0, 1.0), 'y': (-1.0, 1.0)}
# coefficients, given as list: [-0.8, -0.2, -0.0, 1.0, -0.8, 0.8]
# coefsToVars: {1: '0', 2: '1', 3: '0*0', 4: '0*1', 5: '1*1'}
# programVars: ['x', 'y']
def getInvariantVolumeMC(ranges, coefficients, coefToVars, programVars, iterations):
    count_in_inv = 0

    for i in range(iterations):
        # sample points from the ranges
        point = []
        for v in programVars:
            lo, hi = ranges[v]
            rndPnt = np.random.default_rng().uniform(lo, hi, 1)[0]  # uniform returns an array
            point.append(rndPnt)
        
        # if point is inside the ellipsoid, count it
        if shape_utils.satisfiesInvariant(point, coefficients, coefToVars, ranges, programVars):
            count_in_inv += 1

    # get volume of the box, i.e. multiply the lengths of all sides
    volume_box = 1.0
    for v in programVars:
        lo, hi = ranges[v]
        volume_box = volume_box * (hi - lo)
  
    return volume_box * (count_in_inv / iterations)

# example invariant for filter_goubault
# vol = getInvariantVolumeMC({'x': (-1.5, 1.5), 'y': (-1.5, 1.5)}, [-0.1, 0.0, 0.0, 0.5, -1.0, 0.6],
#   {1: '0', 2: '1', 3: '0*0', 4: '0*1', 5: '1*1'}, ['x', 'y'], 1000000)
# print(f'volume: {vol}')
#
# # Area: 1.4049629462081454
# # Ranges: {'s1': (-1.5, 1.5), 's0': (-1.5, 1.5)}
# # Shape: 0.5s1^2 + -1.0s1s0 + 0.6s0^2 <= 0.1
#
# ellipseArea = shape_utils.ellipse_area([-0.1, 0.0, 0.0, 0.5, -1.0, 0.6])
# print(f'ellipse area: {ellipseArea}')

## Explore values
# n = 100
# cmp = {}
# while n <= 10000000:
#     cmp[n] = []
#     n = n*10
#
# for i in range(5):
#     for n in cmp:
#         vol = getInvariantVolumeMC({'x': (-1,1), 'y': (-1,1)}, [-1,0,0,1,0,1], {1: '0', 2: '1', 3: '0*0', 4: '0*1', 5: '1*1'}, ['x', 'y'], n)
#         print(f'Vol for {n} is {vol}')
#         cmp[n].append(vol)
#
# for n in cmp:
#     print(f'{n}: {cmp[n]}')
