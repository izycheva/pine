import math

from sklearn.preprocessing import PolynomialFeatures
import numpy as np
from numpy import linalg


# checks whether a point satisfies the invariant
def satisfiesInvariant(point, coefficients, coefToVars, ranges, programVars):
    for i, v in enumerate(programVars):
        x = point[i]
        (lo, hi) = ranges[v]
        if x < lo or x > hi:
            return False

    radius = computeRad(coefficients, coefToVars, [point])
    return radius <= -coefficients[0]


# checks whether a point satisfies the precondition (input ranges)
def satisfiesPrecondition(point, ranges, programVars):
    for i, v in enumerate(programVars):
        x = point[i]
        lo, hi = float(ranges[v][0]), float(ranges[v][1])
        if x < lo or x > hi:
            return False
    return True


def guessShape(hullVertices, vars, degree=2):
    poly = PolynomialFeatures(degree)
    features = poly.fit_transform(hullVertices)
    featureList = poly.get_feature_names_out(vars)
    featureToCoef = {}

    for i in range(1, len(featureList)):
        usedList = featureList[i].rsplit()

        # translate each term into indices
        usedListFormatted = []
        for used in usedList:
            if '^' in used:
                varName, power = used.split('^')
                term = '*'.join(([str(vars.index(varName))] * int(power)))
                usedListFormatted.append(term)

            else:  # single var
                usedListFormatted.append(str(vars.index(used)))

        # merge into products
        v = '*'.join(usedListFormatted)

        featureToCoef[i] = v

    # set the last feature to one
    new_features = []
    b = []
    for row in features:
        last_feat = row[-1]
        row_without_last = row[:-1]
        new_features.append(list(row_without_last))
        b.append(-last_feat)

    (coef, residuals, rank, s) = np.linalg.lstsq(new_features, b, rcond=-1)
    coef = np.append(coef, np.array([1.0]))

    return coef, featureToCoef


def getMinVolEllipse(P=None, tolerance=0.01):
    """ Find the minimum volume ellipsoid which holds all the points

    Based on work by Nima Moshtagh
    http://www.mathworks.com/matlabcentral/fileexchange/9542
    and also by looking at:
    http://cctbx.sourceforge.net/current/python/scitbx.math.minimum_covering_ellipsoid.html
    Which is based on the first reference anyway!

    Here, P is a numpy array of N dimensional points like this:
    P = [[x,y,z,...], <-- one point per line
         [x,y,z,...],
         [x,y,z,...]]

    Returns:
    (center, radii, rotation)

    """
    if isinstance(P, list):
        P = np.array(P)
    (N, d) = np.shape(P)
    d = float(d)

    # Q will be our working array
    Q = np.vstack([np.copy(P.T), np.ones(N)])
    QT = Q.T

    # initializations
    err = 1.0 + tolerance
    u = (1.0 / N) * np.ones(N)

    # Khachiyan Algorithm
    while err > tolerance:
        V = np.dot(Q, np.dot(np.diag(u), QT))
        M = np.diag(np.dot(QT, np.dot(linalg.inv(V), Q)))  # M the diagonal vector of an NxN matrix
        j = np.argmax(M)
        maximum = M[j]
        step_size = (maximum - d - 1.0) / ((d + 1.0) * (maximum - 1.0))
        new_u = (1.0 - step_size) * u
        new_u[j] += step_size
        err = np.linalg.norm(new_u - u)
        u = new_u

    # center of the ellipse
    center = np.dot(P.T, u)

    # the A matrix for the ellipse
    # TODO get conic form directly from the matrix
    A = linalg.inv(
        np.dot(P.T, np.dot(np.diag(u), P)) -
        np.array([[a * b for b in center] for a in center])
    ) / d

    # Get the values we'd like to return
    U, s, rotation = linalg.svd(A)
    radii = 1.0 / np.sqrt(s)

    return center, radii, rotation


def polarToConic(center, radii, rotation):
    coefficients = []
    # cover 2d case
    if len(center) == 2:
        # rotation matrix must be [[cos(tau) sin(tau)][-sin(tau) cos(tau)]]
        # here looks like it is [[-sin(tau) cos(tau)][cos(tau) sin(tau)]]
        c = rotation[0][0]  # cos(tau)
        s = rotation[0][1]  # sin(tau)
        a, b = radii
        h, k = center
        # Ax^2 + Bxy + Cy^2 + Dx + Ey + F = 0
        # A = (bc)^2  + (as)^2
        A = math.pow(b * c, 2) + math.pow(a * s, 2)
        # B = − 2cs(a^2 − b^2)
        B = -2 * c * s * (a * a - b * b)
        # C = (bs)^2 + (ac)^2
        C = math.pow(b * s, 2) + math.pow(a * c, 2)
        # D = -2Ah -kB
        D = -2 * A * h - k * B
        # E = -2Ck - hB
        E = -2 * C * k - h * B
        # F = -(ab)^2 + Ah^2 + Bhk + Ck^2
        F = -(a * b * a * b) + A * h * h + B * h * k + C * k * k
        coefficients.append(F)
        coefficients.append(D)
        coefficients.append(E)
        coefficients.append(A)
        coefficients.append(B)
        coefficients.append(C)
    else:
        raise Exception("Can't convert higher dimensions of ellipsoids to conic form yet")
    return coefficients


def getMinEllipsoidCoefficients(points, vars={}, precision=0.01):
    center, radii, rotation = getMinVolEllipse(points, precision)
    coefsToVars = {}
    if len(vars) == 2:
        coefsToVars[1] = '0'
        coefsToVars[2] = '1'
        coefsToVars[3] = '0*0'
        coefsToVars[4] = '0*1'
        coefsToVars[5] = '1*1'

    return polarToConic(center, radii, rotation), coefsToVars


def computeRad(coef, coefToVars, points):
    values = []
    for p in points:
        val = 0
        for i in range(1, len(coef)):
            if coef[i] == 0:
                continue
            vlist = coefToVars[i].rsplit('*')
            multiple = 1
            for vv in vlist:
                varIndex = int(vv)
                multiple = multiple * p[varIndex]
            val = val + coef[i] * multiple
        # val = x * coef[1] + coef[2] * y + coef[3] * x * x + coef[4] * x * y + coef[5] * y * y
        values.append(val)
    return max(values)


def formatShape(coef, coefToVars, vars):
    terms = []
    for i in range(1, len(coef)):
        # reconstruct factor
        if not coef[i] == 0.0:
            vIndices = coefToVars[i].rsplit('*')
            factors = []
            # count number of occurences
            for j, v in enumerate(vars):
                count = vIndices.count(str(j))
                if count == 1:
                    factors.append(v)
                elif count > 1:
                    factors.append(f'{v}^{count}')
            factor = '*'.join(factors)
            terms.append(f'{coef[i]}*{factor}')
    return ' + '.join(terms) + f' <= {-coef[0]}'


def getSymmetricPts(coefs, pt) -> list:
    """
    Only for 2D points! Return a set of symmetric points wrt symmetry axes of the shape (ellipse)
    :param coefs: list of coefficients describing the shape
    :param pt: original counter example point (tuple or list of 2 elements)
    :return: set of symmetric points wrt symmetry axes in ellipse
    """

    # from https://stackoverflow.com/questions/49061521/projection-of-a-point-to-a-line-segment-python-shapely
    def getPt(pt, start, end):
        x = np.array(pt)

        u = np.array(start)
        v = np.array(end)

        n = v - u
        n /= np.linalg.norm(n, 2)

        P = u + n * np.dot(x - u, n)
        P2 = P + P - x

        return P2

    if len(coefs) != 6:  # if not ellipse
        return []

    pts = list()

    a = coefs[3]
    b = coefs[4]
    c = coefs[5]
    d = coefs[1]
    e = coefs[2]

    def symmetryAxisPlus(x):
        # TODO check that b^2 - 4ac <> 0 (find out what to do otherwise)
        return (c - a + math.sqrt((a - c) * (a - c) + b * b)) / b * (x - (2 * c * d - b * e) / (b * b - 4 * a * c)) + (
                2 * a * e - b * d) / (b * b - 4 * a * c)

    def symmetryAxisMinus(x):
        return (c - a - math.sqrt((a - c) * (a - c) + b * b)) / b * (x - (2 * c * d - b * e) / (b * b - 4 * a * c)) + (
                2 * a * e - b * d) / (b * b - 4 * a * c)

    if b == 0:
        if d == 0 and e == 0:
            # circle with the center (0,0)
            pts.append([-pt[0], pt[1]])
            pts.append([-pt[0], -pt[1]])
            pts.append([pt[0], -pt[1]])
        else:
            # TODO find a center -> its coordinates (xc,yc) provide the symmetry axes x=xc, y=yc
            return []
    else:
        if b * b - 4 * a * c == 0:
            # todo find another formula
            return pts

        pm = getPt(pt, (0, symmetryAxisMinus(0)), (1, symmetryAxisMinus(1)))
        pts.append(pm)

        pp = getPt(pt, (0, symmetryAxisPlus(0)), (1, symmetryAxisPlus(1)))
        pts.append(pp)

        pN = getPt(pp, (0, symmetryAxisMinus(0)), (1, symmetryAxisMinus(1)))
        pts.append(pN)
        pts = [list(t) for t in pts]

    return pts


def ellipse_area(coefs):
    if len(coefs) != 6:
        return 0  # todo think of a better default value?
    a = coefs[3]
    b = coefs[4]
    c = coefs[5]
    d = coefs[1]
    e = coefs[2]
    f = coefs[0]

    if 4*a*c - b*b == 0:
        return 0  # avoid division by zero

    nom = abs(2*math.pi*(a*e*e + c*d*d + d*e*b - f*(4*a*c - b*b)))
    denom = abs((4*a*c - b*b)*math.sqrt(4*a*c - b*b))


    # n1 =math.pi * d*d/(4*a) + (e*e)/(4*c)-f
    # d1 = math.sqrt(a*c)
    # n1 = 2*math.pi
    # d1 = math.sqrt(4*a*c - b*b)
    # return nom/denom, n1/d1
    return nom/denom

# coefs = [2,0,0,1,0,1]
# a,b = ellipse_area(coefs)
# print(a)
# print(b)
