import ast
import copy
import os
import re
import signal
import subprocess
import sys
from time import time

import matplotlib.pyplot as plt
import numpy as np
from scipy.spatial import ConvexHull
import math
from scipy.spatial.qhull import QhullError 
# for versions of scipy >=1.9 
# from scipy.spatial import QhullError

# our libraries
from solver_wrapper import Solver
import plotting_utils
import preprocessing_utils
import smt_check_real  # use smt_check if you want the float solver
import shape_utils


def inc_by_one(n, decimals=0):
    multiplier = 10 ** decimals
    add = 1 if n >= 0 else -1
    return (math.ceil(n * multiplier) + add) / multiplier


def round_up(n, decimals=0):
    multiplier = 10 ** decimals
    return math.ceil(n * multiplier) / multiplier


def round_down(n, decimals=0):
    multiplier = 10 ** decimals
    return math.floor(n * multiplier) / multiplier


def handler(signum, frame):
    raise smt_check_real.TimeOutException


# Returns the points that are dist apart (along each axis from point)
def pointsNearby(point, distFactor, ranges, vars, numNondet=0):
    nearby = [point]
    for i in range(len(point) - numNondet):
        nearbyLo = copy.deepcopy(nearby)
        for p in nearbyLo:
            loR, hiR = ranges[vars[i]]
            dist = distFactor * abs(hiR - loR)
            p[i] = p[i] - dist
        nearbyHi = copy.deepcopy(nearby)
        for p in nearbyHi:
            loR, hiR = ranges[vars[i]]
            dist = distFactor * abs(hiR - loR)
            p[i] = p[i] + dist
        nearby = nearby + nearbyLo
        nearby = nearby + nearbyHi
    return nearby


def pointList2Dict(pnts, progVars):
    res = {progVars[0]: [], progVars[1]: []}
    if not pnts:
        return res
    for i, v in enumerate(progVars, start=0):
        for p in pnts:
            res[v].append(p[i])
    return res


def debug(text, enable=False):
    if enable:
        print(text)




# points - pts on which we compute shape
# all_points - just all points, not valid for computations
# new_points - all pts deduced from counterexamples
# cex_points - cex from solver
# sympoints - symmetric points of the cex
# ------------- main --------------
def get_real_invariant(argv, parsedProgram={}, printIntermediate=False, recompute=False, plotInterm=False, plotFin=False):
    debug(f'Calling real invariant synthesizer with parameters: {argv}', printIntermediate)
    fileName = argv[0]

    ######## Parameter ########
    # Default: 100 1000 0 0.5 500 1 0  100 1000 0 0.5 500 1 0
    decimalCoeffs = int(argv[1]) if len(argv) > 1 else 2  # precision of coefficients
    decimalRadius = int(argv[2]) if len(argv) > 2 else 2  # precision of ellipsoid volume (radius)
    decimalRanges = int(argv[3]) if len(argv) > 3 else 1  # precision of individual var ranges
    m = int(argv[4]) if len(argv) > 4 else 100  # num. random inputs
    n = int(argv[5]) if len(argv) > 5 else 1000  # num. iterations
    numCex = int(argv[6]) if len(argv) > 6 else 0  # number of additional counter examples returned by the solver
    blockFactor = float(argv[7]) if len(argv) > 7 else 0.5  # portion of the interval width to be blocked for cex
    cexIterations = int(argv[8]) if len(argv) > 8 else 500  # number of loop iterations for cex
    symmetricPtsOn = int(argv[9]) == 1 if len(argv) > 9 else True  # enable symmetric points [0,1]
    nearbyPtsOn = int(argv[10]) == 1 if len(argv) > 10 else False  # enable nearby points [0,1]

    plotIntermediate = plotInterm  #False # by default do not plot
    plotFinal = plotFin

    program = parsedProgram if parsedProgram else preprocessing_utils.parse(fileName)

    # global stuff we need
    compiledLoopStmts = preprocessing_utils.getCompiledLoopStatements(program)
    programVars = program['vars']
    nondet = {v: (program['inputRanges'][v][0], program['inputRanges'][v][1]) for v in program['nondet']}

    ######## Get initial set of points through simulation  ########
    # simulate for m random inputs with n iterations
    samples = preprocessing_utils.simulate(program, m, n, fileName)

    try:
        pointsMatrix = np.column_stack(list(samples.values()))
        # signal.signal(signal.SIGALRM, handler)
        # signal.alarm(30)
        hull = ConvexHull(pointsMatrix, qhull_options='QJ0.04')  # option fixes the QHull Error if all points are on one line
        hullVertices = [pointsMatrix[i] for i in hull.vertices]
        hullVertices = [list(t) for t in hullVertices]  # remove the arrays
        # signal.alarm(0)
    except QhullError as qe:
        return {}, f'Could not construct convex hull. Try different input ranges.{qe}', [], 0, program, {}
    except smt_check_real.TimeOutException:
        return {}, 'Timed out constructing convex hull.', [], 0, program, {}

    # plot the new points
    if plotIntermediate and len(programVars) == 2:
        plotting_utils.plotPoints(program, samples, colour='#8dd7bf')
        plotting_utils.plotHull(hullVertices)
        plt.show()

    # points: points considered in each iteration for computation of
    # shape, range and radius
    # type: # list(points), where each point is a list
    points = hullVertices

    # keep track of all the points we have used
    all_points = points
    coefficients = []

    loopcount = 0
    slv = Solver('Z3')
    while loopcount < 100:  # random limit
        try:
            loopcount = loopcount + 1
            debug(f'\nStarting iteration = {loopcount}', printIntermediate)

            ######## Guess shape ########

            try:
                new_coefficients, coefsToVars = shape_utils.getMinEllipsoidCoefficients(points, programVars)
            except Exception:  # or np.linalg.LinAlgError:
                new_coefficients, coefsToVars = shape_utils.guessShape(points, programVars, degree=2)

            debug(f'new_coefficients: {new_coefficients}', printIntermediate)

            if new_coefficients[-1] < 0.0:  # the shape is 'upside down', so we flip it
                new_coefficients = list(map(lambda x: - x, new_coefficients))

            # normalize so that largest coeff is 1.0
            maxCoef = max(np.abs(new_coefficients[1:]))
            new_coefficients = list(map(lambda x: x / maxCoef, new_coefficients))

            # round the coefficients
            coefficients = list(map(lambda x: round(x, decimalCoeffs), new_coefficients))
            debug(f'rounded coefficients: {coefficients}', printIntermediate)

            # get radius
            radius = shape_utils.computeRad(coefficients, coefsToVars, points)
            coefficients[0] = - round_up(radius, decimalRadius)

            ######## Get ranges ########
            pointsXY = [list(t) for t in zip(*points)]  # get x and y coordinates separately

            ranges = {}
            for i, v in enumerate(programVars, start=0):
                debug(f'before rounding: {min(pointsXY[i])} leq {v} leq {max(pointsXY[i])}',printIntermediate)
                lo = round_down(min(pointsXY[i]), decimalRanges)
                up = round_up(max(pointsXY[i]), decimalRanges)
                # assumes that 'points' always has all the points on the hull
                ranges[v] = (lo, up)
            debug(f'ranges: {ranges}', printIntermediate)

            ######## Check invariant ########
            debug("Candidate invariant:", printIntermediate)
            debug(f'\tRanges: {ranges}', printIntermediate)
            debug(f'\tShape: {shape_utils.formatShape(coefficients, coefsToVars, programVars)}', printIntermediate)
            if plotIntermediate and len(programVars) == 2:
                # f, ax = plt.subplots()
                plotting_utils.plotRanges(ranges, programVars, color='green')
                plotting_utils.plotPoints(program, samples, colour='#8dd7bf')
                # plotting_utils.plotPoints(program, pointList2Dict(all_points, programVars), colour='green')
                plotting_utils.plotContour(new_coefficients, coefsToVars, ranges[programVars[0]],
                                           ranges[programVars[1]], colour='navy', linestyle='dashed')
                plotting_utils.plotContour(coefficients, coefsToVars, ranges[programVars[0]], ranges[programVars[1]],
                                           colour='teal')
                # if loopcount == 1:
                #     f.savefig("nonlin1_candidate1.pdf", bbox_inches='tight',rasterized=True)
                # elif loopcount == 2:
                #     plotting_utils.plotPoints(program, pointList2Dict(all_points, programVars), colour='#eb4034')
                #     f.savefig("nonlin1_candidate2.pdf", bbox_inches='tight',rasterized=True)
                # else:
                plotting_utils.plotPoints(program, pointList2Dict(all_points, programVars), colour='#eb4034')
                plt.title(f'Candidate invariant at iteration {loopcount}')
                plt.show()
            # need all parameters for recomputation
            argv = [fileName, decimalCoeffs, decimalRadius, decimalRanges, m, n, numCex, blockFactor, cexIterations, symmetricPtsOn, nearbyPtsOn]
            float_program = addRoundoff(program, ranges, argv) if recompute else program
            cexs, initConditionFailed = smt_check_real.checkInvariantWithCex(slv, float_program, ranges, coefficients,
                                                                             programVars,
                                                                             coefsToVars, numCex, blockFactor,
                                                                             printIntermediate)
            debug(f'cex: {cexs}', printIntermediate)

            if not cexs:
                debug(f"Obtained invariant for precision {decimalCoeffs} {decimalRadius} {decimalRanges}:",printIntermediate)
                debug(f'\tRanges: {ranges}',printIntermediate)
                debug(f'\tShape: {shape_utils.formatShape(coefficients, coefsToVars, programVars)}',printIntermediate)
                # plot obtained invariant
                if len(programVars) == 2 and plotFinal:
                    plotting_utils.plotRanges(ranges, programVars)
                    plotting_utils.plotPoints(program, pointList2Dict(all_points, programVars))
                    plotting_utils.plotContour(coefficients, coefsToVars, ranges[programVars[0]],
                                               ranges[programVars[1]], colour='teal')
                    plt.title(f'Obtained invariant at iteration {loopcount}')
                    plt.show()
                return ranges, shape_utils.formatShape(coefficients, coefsToVars,
                                                       programVars), coefficients, loopcount, program, coefsToVars

            cex_points = []
            for cex in cexs:
                cexVector = []
                for i in range(0, len(programVars)):
                    cexVector.append(cex[programVars[i]])
                for n in nondet:
                    cexVector.append(cex[n])
                cex_points.append(cexVector)
            debug(f'cex_points: {cex_points}', printIntermediate)

            new_points = cex_points

            if symmetricPtsOn:
                # get symmetric points
                xlabel = program['vars'][0]
                ylabel = program['vars'][1]
                symm_points = []
                for cex in cexs:
                    pt = (cex[xlabel], cex[ylabel])
                    symmPts = shape_utils.getSymmetricPts(coefficients, pt)
                    for p in symmPts:
                        if shape_utils.satisfiesPrecondition(p, program['inputRanges'], programVars) \
                                or shape_utils.satisfiesInvariant(p, coefficients, coefsToVars, ranges,
                                                                  program['vars']):
                            # copy the non-deterministic value for all symmetric points ?
                            for n in nondet:
                                p.append(cex[n])
                            symm_points.append(p)
                    debug(f'Symmetric pts:{symmPts}', printIntermediate)
                new_points = new_points + symm_points

            debug(f'# cex points: {len(new_points)}', printIntermediate)

            # plot the cexs
            if plotIntermediate and len(programVars) == 2:
                # f,ax = plt.subplots()
                plotting_utils.plotRanges(ranges, programVars, color='green')
                plotting_utils.plotPoints(program, pointList2Dict(all_points, programVars))
                plotting_utils.plotContour(coefficients, coefsToVars, ranges[programVars[0]], ranges[programVars[1]],
                                           colour='teal')
                plotcexs = [[p[0], p[1]] for p in cex_points]
                plotting_utils.plotPoints(program, pointList2Dict(plotcexs, programVars), colour='red',size=50)
                if symmetricPtsOn:
                    plotsymm = [[p[0], p[1]] for p in symm_points]
                    plotting_utils.plotPoints(program, pointList2Dict(plotsymm, programVars), colour='purple',size=50)

                # if loopcount == 1:
                #     f.savefig("nonlin1_cex1.pdf", bbox_inches='tight')
                plt.title(f'Counterexamples at iteration {loopcount}')
                plt.show()

            ######## Get new points ########
            # collect all the points for which invariant needs to hold

            if nearbyPtsOn:
                # add points near the cex which satisfy the invariant
                for cex in cex_points:
                    nearbyCex = pointsNearby(cex, blockFactor, ranges, programVars, len(nondet))
                    nearbySat = list(filter(
                        lambda point: shape_utils.satisfiesInvariant(point, coefficients, coefsToVars, ranges,
                                                                     programVars),
                        nearbyCex))
                    new_points = new_points + nearbySat
                    debug(f'Nearby pts:{nearbyCex}', printIntermediate)

            # run the current points for a few (j) loop iterations
            loopPoints = new_points
            for j in range(cexIterations):
                res = preprocessing_utils.evalLoopOnce(compiledLoopStmts, programVars, loopPoints, nondet)
                new_points = new_points + res
                loopPoints = [p[:len(programVars)] for p in res]

            debug(f'total new points: {len(new_points)}', printIntermediate)
            rmNondet = [p[:len(programVars)] for p in new_points]
            new_points = rmNondet

            # get convex hull from new points
            try:
                pointsMatrix = points + new_points
                hull = ConvexHull(pointsMatrix,
                                  qhull_options='QJ')  # option fixes the QHull Error if all points are on one line
                hullVertices = [pointsMatrix[i] for i in hull.vertices]
                hullVertices = [list(t) for t in hullVertices]  # remove the arrays
            except ValueError as ve:
                debug(f'ERROR: {ve}',printIntermediate)
                return {}, f'ERROR: {repr(ve).replace(",",";")}', [], loopcount, program, coefsToVars

            all_points = all_points + new_points

            # plot the new points
            if plotIntermediate and len(programVars) == 2:
                plotting_utils.plotPoints(program, pointList2Dict(points, programVars), colour='navy')
                plotting_utils.plotPoints(program, pointList2Dict(new_points, programVars), colour='steelblue')
                plotting_utils.plotHull(hullVertices)
                plt.title(f'New points at iteration {loopcount}')
                plt.show()

            # points to consider in next loop iteration
            points = hullVertices
        except Exception as e:
            debug(f'ERROR: {e}',printIntermediate)
            return {}, str(f'ERROR: {repr(e).replace(",",";")}'), [], loopcount, program, coefsToVars
        except RuntimeWarning as rw:
            return {}, str(f'ERROR: {repr(rw).replace(",",";")}'), [], loopcount, program, coefsToVars
    print(f'No invariant found for precision {decimalCoeffs} {decimalRadius} {decimalRanges}')
    return {}, '', [], loopcount, program, coefsToVars


def findDaisy(path, name='daisy'):
    for root, dirs, files in os.walk(path):
        if name in files:
            return os.path.join(root, name)


def get_fp_invariant(argv, forcePrecision=False, debugInfo=False, plotFin=False, plotInterm=False):
    rangesBench, formattInv, coefficientsBench = {}, '', []
    algIterations, program, coefsToVars = 0, {}, {}
    fileName = argv[1]
    # assume input parameters do not include precision
    otherParams = argv[2:] if not forcePrecision else argv[5:]
    precisions = []
    timeReal = 0

    if forcePrecision:
        fp_start = time()
        # for parameter evaluation only
        precisions = [int(x) for x in argv[2:5]]
        argv = [fileName] + precisions + otherParams
        rangesBench, formattInv, coefficientsBench, algIterations, program, coefsToVars = get_real_invariant(argv,
                                                                                                             printIntermediate=debugInfo, plotFin=plotFin, plotInterm=plotInterm)
        fp_end = time()
        timeReal = fp_end - fp_start
    else:
        fp_start = time()
        for decimalCoeffs, decimalRadius, decimalRanges in zip(range(1, 4), range(1, 4), range(0, 3)):
            precisions = [decimalCoeffs, decimalRadius, decimalRanges]
            # save previous values
            prevRanges, prevInvariant, prevCoefficients = rangesBench, formattInv, coefficientsBench
            prevAlgIterations, prevProgram, prevCoefsToVars = algIterations, program, coefsToVars

            argv = [fileName] + precisions + otherParams  # keep other parameters as well
            rangesBench, formattInv, coefficientsBench, algIterations, program, coefsToVars = get_real_invariant(argv,
                                                                                                                 parsedProgram=program,printIntermediate=debugInfo)  # if program is not parsed yet, it will be inside get_real_inv
            if prevRanges and not rangesBench:
                # too much precision, assign previous values and break the loop
                rangesBench, formattInv, coefficientsBench = prevRanges, prevInvariant, prevCoefficients
                algIterations, program, coefsToVars = prevAlgIterations, prevProgram, prevCoefsToVars
                precisions = [x - 1 for x in precisions]
                break
        fp_end = time()
        timeReal = fp_end - fp_start

    if not rangesBench:
        return rangesBench, formattInv, coefficientsBench, algIterations, 'FailReal', timeReal, 0

    fp_program = addRoundoff(program, rangesBench, [fileName] + precisions + otherParams)

    debug('\nModified program:', debugInfo)
    debug(fp_program['loopBody'], debugInfo)
    debug(fp_program['inputRanges'], debugInfo)
    fp_start = time()
    try:
        # check that obtained invariant still holds
        slv = Solver('Z3')
        cexs, initConditionFailed = smt_check_real.checkInvariantWithCex(slv, fp_program, rangesBench,
                                                                         coefficientsBench,
                                                                         fp_program['vars'],
                                                                         coefsToVars, 0, 0,
                                                                         False)
        fp_endConfirmed = time()
    except smt_check_real.TimeOutException:
        fp_endTimeout = time()
        print("Solver timed out checking FP invariant")
        return rangesBench, 'Solver timed out checking FP invariant', coefficientsBench, algIterations, 'Timeout', timeReal, fp_endTimeout - fp_start

    if cexs:
        print('Invariant not confirmed for floating-point loop. Giving it another try...')
        debug(cexs, debugInfo)
        argv = [fileName] + precisions + otherParams
        rangesBench, formattInv, coefficientsBench, algIterations, program, coefsToVars = get_real_invariant(
            argv, recompute=True)
        fp_endRecompute = time()
        if rangesBench:
            print('Now we\'re talking')
            print(f'\t{formattInv}')
            print(f'\t{rangesBench}')
            return rangesBench, formattInv, coefficientsBench, algIterations, 'RecomputeFP', timeReal, fp_endRecompute - fp_start
        else:
            print('Fail. Cannot find a FP invariant.')
            return rangesBench, formattInv, coefficientsBench, algIterations, 'FailFP', timeReal, fp_endRecompute - fp_start
    else:
        debug("Invariant confirmed for floating-point implementation:", debugInfo)
        debug(f'\t{formattInv}', debugInfo)
        debug(f'\t{rangesBench}', debugInfo)
        return rangesBench, formattInv, coefficientsBench, algIterations, 'OK', timeReal, fp_endConfirmed - fp_start


def addRoundoff(program, ranges, parameters):
    def parenthesized(expr):
        # enclose each subexpression in parentheses
        tree = ast.parse(expr)
        visitor = preprocessing_utils.ParenthesisVisitor()  # when we can use diff statements, assignedVars)
        visitor.visit(tree)
        assignment = visitor.withParenthesis
        return assignment

    def is_number(s):
        try:
            float(s)
            return True
        except ValueError:
            return False

    def get_error(expr):
        # write necessary info to the temp file
        tmpFile = open(tmpFileName, 'w')
        tmpFile.write(textRanges)
        # enclose each subexpression in parentheses
        assignment = parenthesized(expr)

        tmpFile.write('\n' + assignment)
        tmpFile.close()

        # call Daisy to get roundoff error
        process = subprocess.Popen([daisycmd, tmpFileName, 'Float32'],
                                   stdout=subprocess.PIPE)
        out_all = process.stdout.readlines()
        for line in out_all:
            out = line.decode("utf-8")
            if 'Error' in out:
                break

        # Parse the rounding error value obtained from Daisy
        try:
            err = out.split('Error:')[1].strip()  # rm \n and Error
            if 'e' in err:
                # round the error to one digit after decimal pt
                rnd = err[:(err.index('e') - len(err))]
                rnd = f'{round_up(float(rnd), 1)}{err[err.index("e"):]}'
            elif float(err) == 0:
                rnd = err
            else:
                rnd_to = re.search(r'[^\.|0]', err).start() - 1
                rnd = str(round_up(float(err), rnd_to))
            return rnd
        except Exception as ex:
            print(f"There was a problem getting a rounding error for the input {assignment} \nCannot continue")
            exit(100)

    # look for daisy script in the current directory
    daisycmd = findDaisy(os.getcwd())
    fileName = parameters[0]
    precisions = parameters[1:4]
    otherParams = parameters[4:]
    benchmark_name = fileName.split('/')[-1][:-4]
    str_argv = [str(x) for x in precisions + otherParams]
    tmpFileName = f'tmp_{"_".join([benchmark_name] + str_argv)}.txt'

    # make a separate copy of the program
    fp_program = copy.deepcopy(program)

    # for every branch and every variable assignment compute the roundoff error
    textRanges = ""
    for rv in ranges:
        textRanges = f'{textRanges}{rv} @ [{ranges[rv][0]}, {ranges[rv][1]}]\n'
    for nd in program['nondet']:
        textRanges = f'{textRanges}{nd} @ [{program["inputRanges"][nd][0]}, {program["inputRanges"][nd][1]}]\n'

    # todo handle branches combined with 'true' branch
    branch_mapping = {b:'' for b in program['loopBody']}
    cacheErrs = {}  # expr: term (ranges are everywhere the same, we don't improve on conditions)
    for i, branch in enumerate(program['loopBody']):
        if branch.startswith('not'):
            then = branch[4:]
            branch_existing = f'not {branch_mapping[then]}'
        else:
            branch_existing = branch

        for v in program['loopBody'][branch]:
            stmt = program['loopBody'][branch][v]
            # if we have already computed the error for this expression, simply add the error term, do not recompute
            if parenthesized(stmt) in cacheErrs:
                fp_program['loopBody'][branch_existing][v] = f'{stmt} + {cacheErrs[parenthesized(stmt)]}'
                continue

            if stmt in program['vars'] or is_number(stmt):
                continue

            rnd = get_error(stmt)
            if float(rnd) != 0:
                # add a non-deterministic term for the error
                fp_program['inputRanges'][f'rnd{v}{i}'] = (float(f'-{rnd}'), float(rnd))
                fp_program['loopBody'][branch_existing][v] = f'{stmt} + rnd{v}{i}'
                fp_program['nondet'].append(f'rnd{v}{i}')
                cacheErrs[parenthesized(stmt)] = f'rnd{v}{i}'

        # roundoff on conditions
        # root and else branches do not have roundoff
        if branch == 'true' or branch.startswith('not'):
            continue
        # expr = branch.split('<=>')
        # get lhs, operator, and rhs
        tree = ast.parse(branch)
        visitor = preprocessing_utils.ComparisonVisitor()  # when we can use diff statements, assignedVars)
        visitor.visit(tree)
        lhs, op, rhs = visitor.exprs  # we assume all conditions are of this shape
        # [ast.dump(x) for x in expr[0]]
        new_branch = ''

        if lhs in program['vars'] or is_number(lhs):
            if rhs in program['vars'] or is_number(rhs):
                # comparison between two vars or var and a number or 2 numbers
                continue
            else:
                # rhs is an expression
                if parenthesized(rhs) in cacheErrs:
                    new_branch = f'{lhs} {op} {rhs} + {cacheErrs[parenthesized(rhs)]}'
                    not_new_branch = f'not {new_branch}'
                    fp_program['loopBody'][new_branch] = fp_program['loopBody'].pop(branch)
                    fp_program['loopBody'][not_new_branch] = fp_program['loopBody'].pop(f'not {branch}')
                    branch_mapping[branch] = new_branch
                else:
                    rnd_rhs = get_error(rhs)
                    if float(rnd_rhs) != 0:
                        new_branch = f'{lhs} {op} {rhs} + rndr{i}{i}'
                        fp_program['inputRanges'][f'rndr{i}{i}'] = (float(f'-{rnd_rhs}'), float(rnd_rhs))
                        fp_program['nondet'].append(f'rndr{i}{i}')
                        cacheErrs[parenthesized(rhs)] = f'rndr{i}{i}'
                        fp_program['loopBody'][new_branch] = fp_program['loopBody'].pop(branch)
                        not_new_branch = f'not {new_branch}'
                        fp_program['loopBody'][not_new_branch] = fp_program['loopBody'].pop(f'not {branch}')
                        branch_mapping[branch] = new_branch
                        # program[new_branch] = program[branch]
                        # del program[branch]

        elif rhs in program['vars'] or is_number(rhs):
            # lhs is an expression, rhs is a number or var
            if parenthesized(lhs) in cacheErrs:
                new_branch = f'{lhs} + {cacheErrs[parenthesized(lhs)]} {op} {rhs}'
                fp_program['loopBody'][new_branch] = fp_program['loopBody'].pop(branch)
                not_new_branch = f'not {new_branch}'
                fp_program['loopBody'][not_new_branch] = fp_program['loopBody'].pop(f'not {branch}')
                branch_mapping[branch] = new_branch
            else:
                rnd_lhs = get_error(lhs)
                if float(rnd_lhs) != 0:
                    new_branch = f'{lhs} + rndl{i}{i} {op} {rhs}'
                    fp_program['inputRanges'][f'rndl{i}{i}'] = (float(f'-{rnd_lhs}'), float(rnd_lhs))
                    fp_program['nondet'].append(f'rndl{i}{i}')
                    cacheErrs[parenthesized(lhs)] = f'rndl{i}{i}'
                    fp_program['loopBody'][new_branch] = fp_program['loopBody'].pop(branch)
                    not_new_branch = f'not {new_branch}'
                    fp_program['loopBody'][not_new_branch] = fp_program['loopBody'].pop(f'not {branch}')
                    branch_mapping[branch] = new_branch
        else:
            # lhs and rhs are expressions
            if parenthesized(lhs) in cacheErrs:
                if parenthesized(rhs) in cacheErrs:
                    new_branch = f'{lhs} + {cacheErrs[parenthesized(lhs)]} {op} {rhs} + {cacheErrs[parenthesized(rhs)]}'
                    fp_program['loopBody'][new_branch] = fp_program['loopBody'].pop(branch)
                    not_new_branch = f'not {new_branch}'
                    fp_program['loopBody'][not_new_branch] = fp_program['loopBody'].pop(f'not {branch}')
                    branch_mapping[branch] = new_branch
                else:
                    rnd_rhs = get_error(rhs)
                    new_branch = f'{lhs} + {cacheErrs[parenthesized(lhs)]} {op} {rhs} + rndr{i}{i}'
                    fp_program['inputRanges'][f'rndr{i}{i}'] = (float(f'-{rnd_rhs}'), float(rnd_rhs))
                    fp_program['nondet'].append(f'rndr{i}{i}')
                    cacheErrs[parenthesized(rhs)] = f'rndr{i}{i}'
                    fp_program['loopBody'][new_branch] = fp_program['loopBody'].pop(branch)
                    not_new_branch = f'not {new_branch}'
                    fp_program['loopBody'][not_new_branch] = fp_program['loopBody'].pop(f'not {branch}')
                    branch_mapping[branch] = new_branch
            elif parenthesized(rhs) in cacheErrs:
                rnd_lhs = get_error(lhs)
                new_branch = f'{lhs} + rndl{i}{i} {op} {rhs} + {cacheErrs[parenthesized(rhs)]}'
                fp_program['inputRanges'][f'rndl{i}{i}'] = (float(f'-{rnd_lhs}'), float(rnd_lhs))
                fp_program['nondet'].append(f'rndl{i}{i}')
                cacheErrs[parenthesized(lhs)] = f'rndl{i}{i}'
                fp_program['loopBody'][new_branch] = fp_program['loopBody'].pop(branch)
                not_new_branch = f'not {new_branch}'
                fp_program['loopBody'][not_new_branch] = fp_program['loopBody'].pop(f'not {branch}')
                branch_mapping[branch] = new_branch
            else:
                rnd_lhs = get_error(lhs)
                fp_program['inputRanges'][f'rndl{i}{i}'] = (float(f'-{rnd_lhs}'), float(rnd_lhs))
                fp_program['nondet'].append(f'rndl{i}{i}')
                cacheErrs[parenthesized(lhs)] = f'rndl{i}{i}'

                rnd_rhs = get_error(rhs)
                fp_program['inputRanges'][f'rndr{i}{i}'] = (float(f'-{rnd_rhs}'), float(rnd_rhs))
                fp_program['nondet'].append(f'rndr{i}{i}')
                cacheErrs[parenthesized(rhs)] = f'rndr{i}{i}'

                new_branch = f'{lhs} + rndl{i}{i} {op} {rhs} + rndr{i}{i}'
                fp_program['loopBody'][new_branch] = fp_program['loopBody'].pop(branch)
                not_new_branch = f'not {new_branch}'
                fp_program['loopBody'][not_new_branch] = fp_program['loopBody'].pop(f'not {branch}')
                branch_mapping[branch] = new_branch

    # delete tmpFile
    os.remove(tmpFileName)
    return fp_program


if __name__ == "__main__":
    np.seterr(all='raise')
    signal.signal(signal.SIGALRM, handler)
    signal.alarm(1200)  # 20 minutes
    res = get_fp_invariant(sys.argv, forcePrecision=True, debugInfo=False, plotInterm=False, plotFin=False)
    print(res)
