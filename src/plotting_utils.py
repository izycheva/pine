import numpy as np
import matplotlib.pyplot as plt
import math


# pretty printing and plotting
def plotPoints(program, points, colour='#1f77b4', size=10):
    if len(program['vars']) == 2:
        (x, y) = program['vars']
        xx = points[x]
        yy = points[y]

        plt.scatter(xx, yy, s=size, c=colour, rasterized=True)
        plt.gca().set_aspect('equal', adjustable='box')


def plotHull(hullVertices, colour='r'):
    refvec = [0, 1]
    origin = [0, 0]  # hullVertices[0]
    if len(origin) != 2:
        return

    # from https://stackoverflow.com/questions/41855695/sorting-list-of-two-dimensional-coordinates-by-clockwise-angle-using-python
    def clockwiseangle_and_distance(point):
        # Vector between point and the origin: v = p - o
        vector = [point[0] - origin[0], point[1] - origin[1]]
        # Length of vector: ||v||
        lenvector = math.hypot(vector[0], vector[1])
        # If length is zero there is no angle
        if lenvector == 0:
            return -math.pi, 0
        # Normalize vector: v/||v||
        normalized = [vector[0] / lenvector, vector[1] / lenvector]
        dotprod = normalized[0] * refvec[0] + normalized[1] * refvec[1]  # x1*x2 + y1*y2
        diffprod = refvec[1] * normalized[0] - refvec[0] * normalized[1]  # x1*y2 - y1*x2
        angle = math.atan2(diffprod, dotprod)
        # Negative angles represent counter-clockwise angles so we need to subtract them
        # from 2*pi (360 degrees)
        if angle < 0:
            return 2 * math.pi + angle, lenvector
        # I return first the angle because that's the primary sorting criterium
        # but if two vectors have the same angle then the shorter distance should come first.
        return angle, lenvector

    sortedV = sorted(hullVertices, key=clockwiseangle_and_distance)
    sortedV.append(sortedV[0])
    lists = [list(t) for t in zip(*sortedV)]
    plt.plot(lists[0], lists[1], f'{colour}--', lw=1)


def plotContour(coefs, coefToVars, rangeX, rangeY, colour='blue', linestyle='solid'):
    # determine min and max values of plot
    minX, maxX, minY, maxY = rangeX[0], rangeX[1], rangeY[0], rangeY[1]
    diffX = (maxX - minX) / 3  # 2.0
    diffY = (maxY - minY) / 3
    minX = minX - diffX
    maxX = maxX + diffX
    minY = minY - diffY
    maxY = maxY + diffY

    # plot the shape
    x = np.linspace(min(-3, minX), max(maxX, 3), 1000)
    y = np.linspace(min(-3, minY), max(maxY, 3), 1000)
    X, Y = np.meshgrid(x, y)

    F = coefs[0]
    for i, coef in enumerate(coefs[1:], 1):
        factor = coefToVars[i]
        term = coef
        for v in factor.split('*'):
            if v == '0':
                term = term * X
            elif v == '1':
                term = term * Y
        F = F + term

    # F = X * coefs[1] + coefs[2] * Y + coefs[3] * X * X + coefs[4] * X * Y + coefs[5] * Y * Y + coefs[0]

    plt.contour(X, Y, F, [0], colors=[colour], linestyles=linestyle)

    plt.xlim(minX, maxX)
    plt.ylim(minY, maxY)

    plt.gca().set_aspect('equal', adjustable='box')


def plot2dPolar(center, radii, rotation):
    u = np.linspace(0.0, 2.0 * np.pi, 100)
    v = np.linspace(0.0, np.pi, 100)

    # cartesian coordinates that correspond to the spherical angles:
    x = radii[0] * np.outer(np.cos(u), np.sin(v))
    y = radii[1] * np.outer(np.sin(u), np.sin(v))

    # rotate accordingly
    for i in range(len(x)):
        for j in range(len(x)):
            [x[i, j], y[i, j]] = np.dot([x[i, j], y[i, j]], rotation) + center
    plt.scatter(x, y)
    plt.show()


def plotRanges(ranges, vars, color='red', filll=False):
    # lower left corner
    xy = (ranges[vars[0]][0], ranges[vars[1]][0])
    width = abs(ranges[vars[0]][1] - ranges[vars[0]][0])
    height = abs(ranges[vars[1]][1] - ranges[vars[1]][0])

    rect = plt.Rectangle(xy, width, height, fill=filll, color=color)
    plt.gca().add_patch(rect)
