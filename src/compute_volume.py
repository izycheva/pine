import sys

import numpy as np

import monte_carlo_volume
import shape_utils
import csv

tool = sys.argv[1]
samples = int(sys.argv[2])

out_dir = 'out/'

writecsv = open(f'{out_dir}{tool}_volumes_recomputed.csv','w')
writecsv.write('Benchmark,Volume\n')
writecsv.close()

with open(f'{out_dir}src_tools/{tool}_invariants.csv', newline='') as csvfile:
    rdr = csv.reader(csvfile)
    lc = 0
    volumes = {}
    for row in rdr:
        writecsv = open(f'{out_dir}{tool}_volumes_recomputed.csv','a')
        cc = []
        programVars = []
        coefsToVars = {}
        if row[1].startswith('none'):
            writecsv.write(f'{row[0]},-\n')
        if lc > 0 and not row[1].startswith('none'):
            # '-16.00 <= y <= 16.00\n-16.00 <= x <= 16.00\n-1.00 <= in0 <= 1.00'
            range_to_parse = row[1].split('\n')
            ranges = {}
            for v in range_to_parse:
                lo, var, hi = v.split('<=')
                lo, hi = float(lo), float(hi)
                var = var.strip()
                programVars.append(var)
                ranges[var] = (lo, hi)

            inv_to_parse = row[2].replace('(','').replace(')','').replace('\n','').split('<=')
            cc.append(-1 * float(inv_to_parse[-1]))
            # parse the polynomial
            terms = inv_to_parse[-2].split('+')
            for i, t in enumerate(terms):
                tt = [x.strip() for x in t.split('*')]
                collect_indices = []
                hasCoef = False
                for m in tt:
                    # each multiple is either a number, a var, or var^2
                    if '^2' in m:
                        var = m[:-2]
                        varInd = programVars.index(var)
                        coefsToVars[i + 1] = f'{varInd}*{varInd}'
                        continue

                    try:
                        # number
                        n = float(m)
                        cc.append(n)
                        hasCoef = True
                    except Exception:
                        # var
                        varInd = programVars.index(m)
                        collect_indices.append(varInd)
                if not hasCoef:
                    cc.append(1.0)

                if collect_indices:
                    coefsToVars[i + 1] = '*'.join([str(x) for x in collect_indices])

            vol = monte_carlo_volume.getInvariantVolumeMC(ranges, cc, coefsToVars, programVars, samples)
            # print(f'{row[0]}: {vol}')
            volumes[row[0]] = vol
            writecsv.write(f'{row[0]},{round(vol,2)}\n')

        lc += 1
        writecsv.close()

for p in volumes:
    print(f'{p}, {volumes[p]}')
# pilat_volumes_recomputed