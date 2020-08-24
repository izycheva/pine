import sys
from os import path
from time import time

import general_script
import monte_carlo_volume


######### Define parameters ############
m = 100     # number of random samples from input range - by default 100
n = 1000    # number of loop iterations for initial simulation - by default 1000
l = 0       # number of additional counterexamples - by default 0
d = 0.5     # distance to nearby points - by default 0.5
k = 500     # number of loop iterations for simulation from counterexamples - by default 500
symPt = 1   # enable symmetric points (possible values: 0, 1) - by default 1
nearbyPt = 0  # enable nearby points (possible values: 0, 1) - by default 0
pc = 2      # precision for the shape coefficients - by default 2
pr = 1      # precision for ranges - by default 1

#########################################


def ctv(inv, progVars):
    cctvv = {}
    inv_to_parse = inv.split('<=')
    # parse the polynomial
    terms = inv_to_parse[0].split('+')
    for i, t in enumerate(terms):
        tt = [x.strip() for x in t.split('*')]
        collect_indices = []
        for m in tt:
            # each multiple is either a number, a var, or var^2
            if '^2' in m:
                var = m[:-2]
                varInd = progVars.index(var)
                cctvv[i + 1] = f'{varInd}*{varInd}'
                continue

            try:
                # number
                nn = float(m)
            except Exception:
                # var
                varInd = progVars.index(m)
                collect_indices.append(varInd)

        if collect_indices:
            cctvv[i + 1] = '*'.join([str(x) for x in collect_indices])
    return cctvv


if not path.exists(f'out/inv_res.csv'):
    f = open(f'out/inv_res.csv', 'w')
    f.write(f'Benchmark,Volume,Time,Status,Ranges,Invariant\n')
    f.close()

src_dir = 'benchmarks/'
b = sys.argv[1]
vol_samples = 1000000 if len(sys.argv) < 3 else sys.argv[2]   # number of samples used to estimate volume - by default 1000000
separate_csv = len(sys.argv) > 3  # write results in a separate csv

# put arguments together
args = ['', f'{src_dir}{b}.txt', pc, pc, pr, m, n, l, d, k, symPt, nearbyPt]

# rangesBenchmark = {}
# coefficientsBenchmark = []

start = time()
# call PyINV to get an invariant
rangesBenchmark, formattedInvariant, coefficientsBenchmark, algIterations, success_status, timeReal, timeFP = general_script.get_fp_invariant(
    args, forcePrecision=True, debugInfo=False)
end = time()

if not coefficientsBenchmark:
    f = open('out/pyinv.log', 'a')
    f.write(
        f"RESULTS for {b}:\nRanges: {rangesBenchmark}\nInvariant: {formattedInvariant}\nTook {algIterations} and {timeReal} for real val inv, {timeFP} for FP inv {success_status}.\n\n")
    f.close()
    f = open(f'out/{b}_res.csv', 'a') if separate_csv else open(f'out/inv_res.csv', 'a')
    # Benchmark,Volume,Time(s),Status,Ranges,Invariant
    f.write(f'{b},-,{end - start},{success_status},-,-\n')
    f.close()
    # continue
    exit(0)

coefsToVars = ctv(formattedInvariant, list(rangesBenchmark.keys()))
coefsToVars[0] = '0'
coefficients = list(filter((lambda x: x != 0.0), coefficientsBenchmark))

vol = monte_carlo_volume.getInvariantVolumeMC(rangesBenchmark, coefficients, coefsToVars, list(rangesBenchmark.keys()),
                                              100)
print(f"RESULTS for {b}")
f = open('out/pyinv.log', 'a')
f.write(
    f"RESULTS for {b}:\nRanges: {rangesBenchmark}\nInvariant: {formattedInvariant}\nTook {algIterations} and {timeReal} for real val inv, {timeFP} for FP inv {success_status}.\n\n")

print(f'Volume: {round(vol,2)}')
print(f'Took {round(timeReal + timeFP,2)} seconds to generate the invariant. Confirmed for FP: {success_status}.')
print(f'Ranges: {rangesBenchmark}')
print(f'Invariant: {formattedInvariant}')
f.close()

f = open(f'out/{b}_res.csv', 'a') if separate_csv else open(f'out/inv_res.csv', 'a')
rangesstr = f'{rangesBenchmark}'.replace(',', ';')
coefsstr = f'{coefficientsBenchmark}'.replace(',', ';').strip('[]')
# Benchmark,Volume,Time(s),Status,Ranges,Invariant
if separate_csv:
    f.write(f'Benchmark,Volume,Time,Status,Ranges,Invariant\n')
f.write(f'{b},{vol},{end - start},{success_status},{rangesstr},{formattedInvariant}\n')
f.close()
