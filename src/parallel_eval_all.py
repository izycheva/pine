import sys, traceback
from os import path

import numpy as np

import general_script
import time


def write_to_csv(fileName, valuesMap, cfgStr):
    # Log resutls to csv
    csvlog = open(fileName, "a")
    for v in valuesMap:
        if not valuesMap[v]:
            valuesMap[v] = '-'
        else:
            valuesMap[v] = str(valuesMap[v])
    add = list(valuesMap.values())
    strr = f"{cfgStr},{','.join(add)}\n"
    csvlog.write(strr)
    csvlog.close()


def writeHeaders(fileName):
    if not path.exists(fileName):
        csvlog = open(fileName, 'a')
        strr = ','.join(row_headers)
        csvlog.write(f'{strr}\n')
        csvlog.close()


b = sys.argv[1] if len(sys.argv) > 1 else 'harmonic.txt'
out = "out/"
bench = b.split('/')[-1]  # benchmark name without path
if b == bench:
    b = f'benchmarks/{b}'
bench = bench[:len(bench) - 4]  # and extension

precisions = [['1', '1', '0'],
              ['2', '2', '1'],
              ['3', '3', '2']]

initSamples = [sys.argv[2]]
numCex = [sys.argv[3]] if len(sys.argv) > 2 else ['0']  # , '1', '2', '5']
distances = [sys.argv[4]]
cexIterations = [sys.argv[5]]
tmp = sys.argv[6].split('_')
symmetricPts = [tmp[0]]
nearbyPts = [tmp[1]]

ranges = {}
formattedInv = {}
coefficients = {}
numIters = {}
times = {}
np.seterr(all='raise')

row_headers = ['Config'] + [' '.join(p) for p in precisions]
writeHeaders(f"{out}csv/{bench}_eval_ranges_{numCex[0]}.csv")
writeHeaders(f"{out}csv/{bench}_eval_invars_{numCex[0]}.csv")
writeHeaders(f"{out}csv/{bench}_eval_fp_confirmed_{numCex[0]}.csv")
writeHeaders(f"{out}csv/{bench}_eval_times_{numCex[0]}.csv")
writeHeaders(f"{out}csv/{bench}_eval_coefs_{numCex[0]}.csv")
writeHeaders(f"{out}csv/{bench}_eval_algiters_{numCex[0]}.csv")
writeHeaders(f"{out}csv/{bench}_eval_time_success_{numCex[0]}.csv")

try:
    for cexs in numCex:
        for i, dist in enumerate(distances):
            for nb in nearbyPts:
                if cexs == '0' and i > 0 and nb == '0':  # for 0 additional counter examples distance does not affect anything
                    continue
                for insampl in initSamples:
                    for cexloop in cexIterations:
                        for sym in symmetricPts:
                            time_csv = {' '.join(p): '' for p in precisions}
                            ranges_csv = {' '.join(p): '' for p in precisions}
                            formattedInv_csv = {' '.join(p): '' for p in precisions}
                            coefficients_csv = {' '.join(p): '' for p in precisions}
                            algIters_csv = {' '.join(p): '' for p in precisions}
                            success_csv = {' '.join(p): '' for p in precisions}
                            ins = insampl.split('_')
                            for p in precisions:
                                #     ######## Parameter ########
                                print(
                                    '\n\n============================================================================================')
                                print(
                                    f'============={b} with {p[0]}, {p[1]}, {p[2]}, {ins[0]}, {ins[1]}, {cexs}, {dist}, {cexloop}, {sym}, {nb}=============')
                                #  ORDER OF ARGUMENTS!! -- changed on May 2 -- upd rm p[0], p[1], p[2] or forcePrecision=True

                                args = ['', b, p[0], p[1], p[2], ins[0], ins[1], cexs, dist, cexloop, sym, nb]
                                # ranges, str invariant, coefficients
                                start = time.time()

                                # rangesBenchmark, formattedInvariant, coefficientsBenchmark, algIterations, success_status, timeReal, timeFP = {}, '', [], 0, 'OK', 1, 0.1
                                rangesBenchmark, formattedInvariant, coefficientsBenchmark, algIterations, success_status, timeReal, timeFP = general_script.get_fp_invariant(
                                    args, forcePrecision=True)
                                end = time.time()
                                ranges[
                                    f'{b}_{ins[0]}, {ins[1]}, {p[0]}, {p[1]}, {p[2]}, {cexs}, {dist}, {cexloop}, {sym}, {nb}'] = rangesBenchmark
                                formattedInv[
                                    f'{b}_{ins[0]}, {ins[1]}, {p[0]}, {p[1]}, {p[2]}, {cexs}, {dist}, {cexloop}, {sym}, {nb}'] = formattedInvariant
                                coefficients[
                                    f'{b}_{ins[0]}, {ins[1]}, {p[0]}, {p[1]}, {p[2]}, {cexs}, {dist}, {cexloop}, {sym}, {nb}'] = coefficientsBenchmark
                                numIters[
                                    f'{b}_{ins[0]}, {ins[1]}, {p[0]}, {p[1]}, {p[2]}, {cexs}, {dist}, {cexloop}, {sym}, {nb}'] = algIterations
                                times[
                                    f'{b}_{ins[0]}, {ins[1]}, {p[0]}, {p[1]}, {p[2]}, {cexs}, {dist}, {cexloop}, {sym}, {nb}'] = f'{timeReal};{timeFP};{end - start}'

                                ranges_csv[' '.join(p)] = f'{rangesBenchmark}'.replace(',', ';')
                                formattedInv_csv[' '.join(p)] = formattedInvariant
                                coefficients_csv[' '.join(p)] = ';'.join([str(x) for x in coefficientsBenchmark])
                                algIters_csv[' '.join(p)] = algIterations
                                success_csv[' '.join(p)] = success_status
                                time_csv[' '.join(p)] = f'{timeReal};{timeFP};{end - start}'

                                f = open(f"{out}{bench}_eval_full_{numCex[0]}.log",
                                         "a")  # log inside the loop to avoid loosing data
                                compose = f"\n\n====== Ranges for {b} with parameters {p[0]}, {p[1]}, {p[2]}, {ins[0]}, {ins[1]}, {cexs}, {dist}, {cexloop}, {sym}, {nb} ======\n"
                                for v in rangesBenchmark:
                                    compose = compose + f' {v}: {rangesBenchmark[v]} '
                                compose = compose + f"\n\tFormatted Invariant:{formattedInvariant}"
                                coefficientsBenchmark = [str(x) for x in coefficientsBenchmark]
                                compose = compose + f"\n\tCoefficients: {' '.join(coefficientsBenchmark)}\n"
                                timez = f'\nInvariant found in {algIterations} iterations which took {end - start}s\n' if rangesBenchmark else f'\nInvariant not found after {algIterations} iterations, spent {end - start}s\n'
                                compose = compose + timez
                                compose = compose + f'\nWas the invariant recomputed? {success_status}\n'
                                f.write(compose)
                                f.close()
                            # Log to csv
                            cfg = f'{ins[0]} {ins[1]} {cexs} {dist} {cexloop} {sym} {nb}'
                            write_to_csv(f"{out}csv/{bench}_eval_ranges_{numCex[0]}.csv", ranges_csv, cfg)
                            write_to_csv(f"{out}csv/{bench}_eval_invars_{numCex[0]}.csv", formattedInv_csv, cfg)
                            write_to_csv(f"{out}csv/{bench}_eval_fp_confirmed_{numCex[0]}.csv", success_csv, cfg)
                            write_to_csv(f"{out}csv/{bench}_eval_times_{numCex[0]}.csv", time_csv, cfg)
                            write_to_csv(f"{out}csv/{bench}_eval_coefs_{numCex[0]}.csv", coefficients_csv, cfg)
                            write_to_csv(f"{out}csv/{bench}_eval_algiters_{numCex[0]}.csv", algIters_csv, cfg)
                            # times _ success: only when the invariant is found
                            times_success_csv = {' '.join(p): '-' for p in precisions}
                            for r in ranges_csv:
                                if ranges_csv[r] != '{}':  # ranges map is not empty
                                    times_success_csv[r] = time_csv[r]
                            write_to_csv(f"{out}csv/{bench}_eval_time_success_{numCex[0]}.csv", times_success_csv, cfg)

    summary = open(f"{out}{bench}_summary_{numCex[0]}.log", "a")
    summ = "====== Ranges ======\n"
    for r in ranges:
        summ = summ + f'{r}: {ranges[r]}\n'
    summ = summ + "====== Formatted Invariants ======\n"
    for inv in formattedInv:
        summ = summ + f'{inv}: {formattedInv[inv]}\n'
    summ = summ + "====== Coefficients ======\n"
    for c in coefficients:
        summ = summ + f'{c}: {coefficients[c]}\n'
    summary.write(summ)
    summary.close()
except Exception as e:
    print(f'What happened? {e}')
    traceback.print_exc()