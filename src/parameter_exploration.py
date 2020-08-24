import ast
import csv
import math
import os
import sys
from os import path
from monte_carlo_volume import getInvariantVolumeMC
import numpy as np
import matplotlib.pyplot as plt

############## Adjust if running from somewhere else than pyinv ##########
src_dir = 'out/csv/'
target_dir = 'out/results_sensitivity/'
#########################################################################

volsamples = 1000000 if len(sys.argv) < 2 else sys.argv[1]
benchmarks = [
    'pendulum_approx',
    'rotation_nondet_small_angle',
    'rotation_nondet_large_angle',
    'nonlin_example1',
    'nonlin_example2',
    'nonlin_example3',
    'harmonic',
    'symplectic',
    'filter_goubault',
    'filter_mine1',
    'filter_mine2',
    'filter_mine2_nondet',
    'pendulum_small'
]

files = {
    'time': 'eval_times',
    'iters': 'eval_algiters',
    'coefs': 'eval_coefs',
    'invars': 'eval_invars',
    'ranges': 'eval_ranges',
    'fp_ok': 'eval_fp_confirmed',
    'timeok': 'eval_time_success'
}
# src_dir = '../../exp_results/indiv_call/csv/'
benchmarks = list(filter((lambda b: path.exists(f'{src_dir}{b}_{files["time"]}_0.csv') and path.exists(f'{src_dir}{b}_{files["time"]}_1.csv') and path.exists(f'{src_dir}{b}_{files["time"]}_2.csv') and path.exists(f'{src_dir}{b}_{files["time"]}_5.csv')), benchmarks))
print('Results exist for the following benchmarks:')
for b in benchmarks:
    print(b)

if len(benchmarks) == 0:
    print('No results to parse yet. Run the sensitivity experiment first (./sensitivity.sh).')
    exit(0)

best_times_cfg = {b: [] for b in benchmarks}
best_iters_cfg = {b: [] for b in benchmarks}
best_ranges_cfg = {b: [] for b in benchmarks}
same_ranges_cfg = {b: [] for b in benchmarks}
failed_cfg = {b: [] for b in benchmarks}
relative_success = {b: {'OK': 0, 'RecomputeFP': 0, 'FailFP': 0, 'FailReal': 0} for b in benchmarks}
global column_names
global num_exps


def merge_results(name):
    onecsv = open(f'{src_dir}{b}_{files[name]}.csv', "a")
    # for i in [0,1,2,5]:
    csv0 = open(f'{src_dir}{b}_{files[name]}_{0}.csv', newline='')
    csv1 = open(f'{src_dir}{b}_{files[name]}_{1}.csv', newline='')
    csv2 = open(f'{src_dir}{b}_{files[name]}_{2}.csv', newline='')
    csv5 = open(f'{src_dir}{b}_{files[name]}_{5}.csv', newline='')

    rows0 = list(csv.reader(csv0))
    onecsv.write(f"{','.join(rows0[0])}\n")
    for row in rows0[1:]:
        onecsv.write(f'{",".join(row)}\n')
    for row in list(csv.reader(csv1))[1:]:
        onecsv.write(f'{",".join(row)}\n')
    for row in list(csv.reader(csv2))[1:]:
        onecsv.write(f'{",".join(row)}\n')
    for row in list(csv.reader(csv5))[1:]:
        onecsv.write(f'{",".join(row)}\n')

    csv0.close()
    csv1.close()
    csv2.close()
    csv5.close()
    onecsv.close()


# merge the csv files for the same benchmark together
for b in benchmarks:
    for k in files:
        if path.exists(f'{src_dir}{b}_{files[k]}.csv'):
            os.remove(f'{src_dir}{b}_{files[k]}.csv')
        merge_results(k)

csv0 = open(f'{src_dir}{benchmarks[0]}_{files["invars"]}.csv', newline='')
rdr = list(csv.reader(csv0))
headers = [r[0] for r in rdr[1:]]
csv0.close()

# Find configurations that work for all benchmarks
cfg_to_success1 = {k: 0 for k in headers}
cfg_to_success2 = {k: 0 for k in headers}
cfg_to_success3 = {k: 0 for k in headers}
for b in benchmarks:
    csv0 = open(f'{src_dir}{b}_{files["invars"]}.csv', newline='')
    rdr = csv.DictReader(csv0)
    rows = list(rdr)[1:]
    # count for each precision separately
    success_110 = False
    success_221 = False
    success_332 = False
    for r in rows:
        cfg = r['Config']
        try:
            success_110 = 'ERROR' not in r['1 1 0'] and 'Solver' not in r['1 1 0'] and r['1 1 0'] != '-'
            success_221 = 'ERROR' not in r['2 2 1'] and 'Solver' not in r['2 2 1'] and r['2 2 1'] != '-'
            success_332 = 'ERROR' not in r['3 3 2'] and 'Solver' not in r['3 3 2'] and r['3 3 2'] != '-'
        except Exception as e:
            print(f'{b} at {cfg}')
        if success_110:
            cfg_to_success1[cfg] += 1
        if success_221:
            cfg_to_success2[cfg] += 1
        if success_332:
            cfg_to_success3[cfg] += 1
    csv0.close()

# print(f'Total of {len(benchmarks)} benchmarks')
successfull_cfgs1 = sorted(cfg_to_success1, key=(lambda x: cfg_to_success1[x]))
print(
    f'For precision (pc=1,pr=0) configs work on max {cfg_to_success1[successfull_cfgs1[-1]]}/{len(benchmarks)} benchmarks.\n\n')

successfull_cfgs3 = sorted(cfg_to_success3, key=(lambda x: cfg_to_success3[x]))
print(
    f'for precision (pc=3,pr=2) configs work on max {cfg_to_success3[successfull_cfgs3[-1]]}/{len(benchmarks)} benchmarks.\n\n')

successfull_cfgs2 = sorted(cfg_to_success2.items(), key=(lambda x: x[1]))
selected_cfg = list(filter(lambda x: x[1] == len(benchmarks), successfull_cfgs2))
print(
    f'\nfor precision (pc=2,pr=1) configs work on max {cfg_to_success2[successfull_cfgs2[-1][0]]}/{len(benchmarks)} benchmarks.\n')
print(f'\n{len(selected_cfg)} cfgs suceeded for all benchmarks (out of {len(successfull_cfgs2)*3} cfgs total)')

print('\n\n=== Successful configurations are logged in the out/results_sensitivity/success_cfgs.csv ====')
okcfgfile = open(f'{target_dir}success_cfgs.csv', 'w')
okcfgfile.write('m,n,l,d,k,symPts,nearbyPts\n')
okcfgfile.close()
for i in selected_cfg:
    clist = ','.join(i[0].split(' '))
    # put in csv
    okcfgfile = open(f'{target_dir}success_cfgs.csv', 'a')
    okcfgfile.write(f'{clist}\n')
    okcfgfile.close()

selected_cfg = [x[0] for x in selected_cfg]

# Figure 4

# count the values
analyzed_cfg = [x.split(' ') for x in selected_cfg]
mn1001k = sum(map(lambda x: x[0] == '100' and x[1] == '1000', analyzed_cfg))
mn1k1k = sum(map(lambda x: x[0] == '1000' and x[1] == '1000', analyzed_cfg))
mn10010k = sum(map(lambda x: x[0] == '100' and x[1] == '10000', analyzed_cfg))
# '100 1000 0 0.1 500 1 0'
l0 = sum(map(lambda x: x[2] == '0', analyzed_cfg))
l1 = sum(map(lambda x: x[2] == '1', analyzed_cfg))
l2 = sum(map(lambda x: x[2] == '2', analyzed_cfg))
l5 = sum(map(lambda x: x[2] == '5', analyzed_cfg))
# '100 1000 0 0.1 500 1 0'
d01 = sum(map(lambda x: x[3] == '0.1', analyzed_cfg))
d025 = sum(map(lambda x: x[3] == '0.25', analyzed_cfg))
d05 = sum(map(lambda x: x[3] == '0.5', analyzed_cfg))
# '100 1000 0 0.1 500 1 0'
k0 = sum(map(lambda x: x[4] == '0', analyzed_cfg))
k100 = sum(map(lambda x: x[4] == '100', analyzed_cfg))
k500 = sum(map(lambda x: x[4] == '500', analyzed_cfg))
# '100 1000 0 0.1 500 1 0'
symon = sum(map(lambda x: x[5] == '0', analyzed_cfg))
symoff = sum(map(lambda x: x[5] == '1', analyzed_cfg))
# '100 1000 0 0.1 500 1 0'
nbon = sum(map(lambda x: x[6] == '0', analyzed_cfg))
nboff = sum(map(lambda x: x[6] == '1', analyzed_cfg))

f, ax = plt.subplots()
N = 7
# precision	(m, n)	l	d	k	symPts	nearbyPts
g1 = (len(selected_cfg), mn1001k, l0, d01, k100, symoff, nboff)  # (m,n) == 100-1000; l = 0; d = 0.1; k = 100; symPts = 0; nbPts = 0
g2 = (0, mn1k1k, l1, d025, k500, symon, nbon)  # (m,n) == 1000-1000; l = 1; d = 0.25; k = 500; symPts = 1; nbPts = 1
g3 = (0, mn10010k, l2, d05, k0, 0, 0)  # (m,n) == 100-10k; l = 2; d = 0.5; k = 0
g4 = (0, 0, l5, 0, 0, 0, 0)  # _ ; l = 5

ind = np.arange(N)  # the x locations for the groups
width = 0.8  # the width of the bars: can also be len(x) sequence

p1 = plt.bar(ind, g1, width)
p2 = plt.bar(ind, g2, width, bottom=g1)
p3 = plt.bar(ind, g3, width, bottom=np.array(g1)+np.array(g2))
p4 = plt.bar(ind, g4, width, bottom=np.array(g1)+np.array(g2)+np.array(g3))

plt.title('Proportion of parameters in successful configurations')
plt.xticks(ind, ('prec', '(m,n)', 'l', 'd', 'k', 'symPts', 'nearbyPts'))
scale = math.ceil(len(analyzed_cfg)/100)*100
plt.yticks(np.arange(0, scale, 20))

for i, r1, r2, r3, r4 in zip(range(7), p1, p2, p3, p4):
    h1 = r1.get_height()
    h2 = r2.get_height()
    h3 = r3.get_height()
    h4 = r4.get_height()
    if h1 > 0:
        text = ''
        if i == 0:
            text = '(2,1)'
        elif i == 1:
            text = '(100,\n1000)'
        elif i == 2:
            text = '0'
        elif i == 3:
            text = '0.1'
        elif i == 4:
            text = '100'
        elif i == 5:
            text = 'X'
        else:  # nboff
            text = 'X'
        plt.text(r1.get_x() + r1.get_width() / 2., h1 / 2., text, ha="center", va="center", color="white", fontsize=10, fontweight="bold")
    if h2 > 0:
        # g2 = (0, mn1k1k, l1, d025, k500, symon, nbon)  # (m,n) == 1000-1000; l = 1; d = 0.25; k = 500; symPts = 1; nbPts = 1
        text = ''
        if i == 1:
            text = '(1000,\n1000)'
        elif i == 2:
            text = '1'
        elif i == 3:
            text = '0.25'
        elif i == 4:
            text = '500'
        elif i == 5:
            text = 'V'
        else:  # nbon
            text = 'V'
        plt.text(r2.get_x() + r2.get_width() / 2., h1 + h2 / 2., text, ha="center", va="center", color="white", fontsize=10, fontweight="bold")
    if h3 > 0:
        # g3 = (0, mn10010k, l2, d05, k0, 0, 0)  # (m,n) == 100-10k; l = 2; d = 0.5; k = 0
        text = ''
        if i == 1:
            text = '(100,\n10k)'
        elif i == 2:
            text = '2'
        elif i == 3:
            text = '0.5'
        elif i == 4:
            text = '0'
        plt.text(r2.get_x() + r2.get_width() / 2., h1 + h2 + h3 / 2., text, ha="center", va="center", color="white", fontsize=10, fontweight="bold")
    if h4 > 0:
        # g4 = (0, 0, l5, 0, 0, 0, 0)
        text = ''
        if i == 2:
            text = '5'
        plt.text(r2.get_x() + r2.get_width() / 2., h1 + h2 + h3 + h4 / 2., text, ha="center", va="center", color="white", fontsize=10, fontweight="bold")

# plt.show()
f.savefig(f"{target_dir}figure4.pdf", bbox_inches='tight')

print('\n\n========== Figure 4 is ready! Type `evince out/results_sensitivity/figure4.pdf` ========== \n\n')

# compute smallest volume
all_volumes = {b: {sc[0]: 0 for sc in selected_cfg} for b in benchmarks}
# best_vol_per_bench = {}
avg_times = {b: [] for b in benchmarks}
num_cfg = len(selected_cfg)

volfile = open(f'{target_dir}volumes.csv', 'w')
volfile.write('Benchmark,' + ','.join(selected_cfg) + '\n')
volfile.close()
tfile = open(f'{target_dir}table5.csv', 'w')
tfile.write('Benchmark,Minimum,Average,Maximum\n')
tfile.close()

best_vol_per_bench = {b: 10000 for b in benchmarks}
vol_per_bench = {b: [] for b in benchmarks}
largest_vol = -1
for b in benchmarks:
    volumes = {}
    timereal = 0
    timeFP = 0
    timeTotal = 0

    with open(f'{src_dir}{b}_{files["coefs"]}.csv', newline='') as csvcoefs:
        #     # get ranges
        with open(f'{src_dir}{b}_{files["ranges"]}.csv', newline='') as csvranges:
            def to_map(rangeString):
                try:
                    rs = rangeString.replace(';',
                                             ',') if rangeString != 'repeated_cfg' and rangeString != '-' and rangeString != '' and rangeString != '    ' else '0'
                    if rs == '0':
                        return {}
                    rangeMap = ast.literal_eval(rs)
                    return rangeMap
                except Exception as e:
                    return {}


            def to_coefs(coefstr):
                try:
                    if coefstr != '-' and coefstr != '' and coefstr != '    ':
                        return [float(co) for co in coefstr.strip('[]').split(';')]
                    else:
                        return []
                except Exception:
                    return []


            volline = f'{b}'
            csvtimes = open(f'{src_dir}{b}_{files["time"]}.csv', newline='')
            rdrtimes = csv.DictReader(csvtimes)
            coefsToVars = {1: '0', 2: '1', 3: '0*0', 4: '0*1', 5: '1*1'}

            rdrranges = csv.DictReader(csvranges)
            rdrcoefs = csv.DictReader(csvcoefs)

            allranges = list(rdrranges)
            allcoefs = list(rdrcoefs)
            alltimes = list(rdrtimes)

            j = 1
            prec = '2 2 1'
            for i, row in enumerate(allranges):
                sc = row['Config']
                if sc not in selected_cfg:
                    continue

                rangge = to_map(row[prec])
                coef = to_coefs(allcoefs[i][prec])

                if not coef or not rangge:
                    continue

                vol = getInvariantVolumeMC(rangge, coef, coefsToVars, list(rangge.keys()), int(volsamples))  # 1000000
                print(f'Config #{j}/{len(selected_cfg)} ({sc}) for benchmark {b}. Volume: {vol}')
                j += 1
                volumes[sc] = vol
                all_volumes[b][sc] = vol
                volline = f'{volline},{vol}'
                # also find average time real/fp/total
                time_list = alltimes[i][prec].split(';')
                r, fp, tot = float(time_list[0]), float(time_list[1]), float(time_list[2])
                timereal += r
                timeFP += fp
                timeTotal += tot

            # record the volumes
            volfile = open(f'{target_dir}volumes.csv', 'a')
            strvols = [str(v) for v in volumes.values()]
            volfile.write(f'{b},{",".join(strvols)}\n')
            volfile.close()

            # min max avg for Table 5

            minn = min(volumes.values())
            maxx = max(volumes.values())
            avg = sum(volumes.values()) / len(volumes)
            tfile = open(f'{target_dir}table5.csv', 'a')
            tfile.write(f'{b},{round(minn,2)},{round(avg,2)},{round(maxx,2)}\n')
            tfile.close()

            if max(volumes.values()) > largest_vol:
                largest_vol = max(volumes.values())

            vol_per_bench[b] = volumes

            csvtimes.close()

with open(f'{target_dir}volumes.csv', newline='') as csvvols:
    rdr = csv.DictReader(csvvols)
    head = set(rdr.fieldnames)

    rows = list(rdr)
    summed_vols = {c: 100000 for c in selected_cfg}

    for c in selected_cfg:
        volumes = []
        for b in range(len(benchmarks)):
            if c not in rows[b]:
                continue
            try:
                volumes.append(float(rows[b][c]))
            except Exception as e:
                print(f'Something weird happened for {c} on {rows[b]["Benchmark"]}')
                continue

        scaled_vol = [x / largest_vol for x in volumes]
        summed_vols[c] = sum(scaled_vol)

    smallest = min(summed_vols.values())
    eps = 0.001
    best_avg_vol = dict(
        filter(lambda x: smallest - eps <= x[1] <= smallest + eps, summed_vols.items()))
    bestbest = best_avg_vol
    print(f'\n\nThe configuration with the smallest volume is: {best_avg_vol}\n\n')

print('Top-5 best cfgs:')
for c in sorted(summed_vols.items(), key=(lambda x: x[1]))[:5]:
    print(f'{"{:<27}".format(c[0])} {c[1]}')


