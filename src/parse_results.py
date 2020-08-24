import csv
import os
from os import path


def avg(l):
    if len(l) > 0:
        return sum(l) / len(l)
    else:
        return -1


benchmarks = [
    'pendulum_approx',
    'rotation_nondet_small_angle',
    'rotation_nondet_large_angle',
    'nonlin_example1',
    'nonlin_example2',
    'nonlin_example3',
    'arrow_hurwicz',
    'harmonic',
    'symplectic',
    'filter_goubault',
    'filter_mine1',
    'filter_mine2',
    'filter_mine2_nondet',
    'pendulum_small',
    'ex1', 'ex1_reset',
    'ex2', 'ex2_reset',
    'ex3_leadlag', 'ex3_reset_leadlag',
    'ex4_gaussian', 'ex4_reset_gaussian',
    'ex5_coupled_mass', 'ex5_reset_coupled_mass',
    'ex6_butterworth', 'ex6_reset_butterworth',
    'ex7_dampened', 'ex7_reset_dampened',
    'ex8_harmonic', 'ex8_reset_harmonic',
]

out_dir = 'out/'

if path.exists(f'{out_dir}avg_res.csv'):
    os.remove(f'{out_dir}avg_res.csv')

if not path.exists(f'{out_dir}inv_res.csv'):
    print('No results to parse yet. Run the script `pyinv_table1.sh` first.')
    exit(0)

# f'{b},{avgvol},{variation},{avgtime}'
volumes_per_bench = {b: [] for b in benchmarks}
times_per_bench = {b: [] for b in benchmarks}
status_per_bench = {b: [] for b in benchmarks}

with open(f'{out_dir}inv_res.csv', newline='') as csvresults:
    rows = list(csv.DictReader(csvresults))
    for r in rows:
        b = r['Benchmark']
        if r['Volume'] != '-':
            vol = float(r['Volume'])
            volumes_per_bench[b].append(vol)

        time = float(r['Time'])
        times_per_bench[b].append(time)
        status_per_bench[b].append(r['Status'])

# Record and print avg results
csvwrite = open(f'{out_dir}avg_res.csv', 'a')
csvwrite.write('Benchmark,Volume,Variation,Time\n')
csvwrite.close()

smtai_res = {b:'' for b in benchmarks}
pilat_res = {b:'' for b in benchmarks}

if path.exists(f'{out_dir}compare_volumes_recomputed.csv'):
    print('Using freshly computed volumes for the state-of-the-art tools (see "out/compare_volumes_recomputed.csv").')
    resf = open(f'{out_dir}compare_volumes_recomputed.csv', newline='')
    res = csv.DictReader(resf)
    for r in res:
        b = r['Benchmark']
        smtai_res[b] = r['SMT-AI']
        pilat_res[b] = r['Pilat']
    resf.close()
elif path.exists(f'{out_dir}src_tools/compare_volumes.csv'):
    print('Using precomputed volumes for the state-of-the-art tools (see out/compare_volumes.csv).')
    resf = open(f'{out_dir}src_tools/compare_volumes.csv', newline='')
    res = csv.DictReader(resf)
    for r in res:
        b = r['Benchmark']
        smtai_res[b] = r['SMT-AI']
        pilat_res[b] = r['Pilat']
    resf.close()
else:
    print('No file with state-of-the-art results exists. Run the script "./other_tools.sh" or recover "out/src_tools/compare_volumes.csv".')
    # Print results
    print('\n\n--------------------------------------------------------------------------------------------------')
    print("|                              | Average |   Variation    |  Average invariant  ||  # invariants |")
    print("|          Benchmark           | volume  |   in volume    |  generation time, s ||  generated    |")
    print('--------------------------------------------------------------------------------------------------')

    for b in benchmarks:
        vols = volumes_per_bench[b]
        avgvol = '-'
        variation = '-'

        if len(vols) > 0:
            minv = min(vols)
            maxv = max(vols)
            avgvol = round(avg(vols), 2)
            variation = round(((maxv - minv) / avgvol) * 100, 2)

        avgtime = round(avg(times_per_bench[b]), 2)
        status = status_per_bench[b].count('OK') + status_per_bench[b].count('RecomputeFP')
        f = open(f'{out_dir}avg_res.csv', 'a')
        # print(f'{b}| {avgvol}|{variation}|"{avgtime}')
        f.write(f'{b},{avgvol},{variation}%,{avgtime}\n')
        f.close()
        print(
            '| ' + "{:<28}".format(b) + ' | ' + "{:>7}".format(str(avgvol)) + ' | ' + "{:>13}%".format(
                str(variation)) + ' | '
            + "{:>19}".format(str(avgtime)) + ' || ' + "{:>11}/4".format(str(status)) + ' |')

    print('--------------------------------------------------------------------------------------------------')
    exit(0)

# Print results with other tools
print('\n\n------------------------------------------------------------------------------------------------------------------------')
print("|                              |          |         || Average |   Variation    |  Average invariant  ||  # invariants |")
print("|          Benchmark           |  SMT-AI  |  Pilat  || volume  |   in volume    |  generation time, s ||  generated    |")
print('------------------------------------------------------------------------------------------------------------------------')
for b in benchmarks:
    vols = volumes_per_bench[b]
    avgvol = '-'
    variation = '-'

    if len(vols) > 0:
        minv = min(vols)
        maxv = max(vols)
        avgvol = round(avg(vols), 2)
        variation = round(((maxv - minv) / avgvol) * 100, 2)

    avgtime = round(avg(times_per_bench[b]), 2)
    status = status_per_bench[b].count('OK') + status_per_bench[b].count('RecomputeFP')
    f = open(f'{out_dir}avg_res.csv', 'a')
    # print(f'{b}| {avgvol}|{variation}|"{avgtime}')
    f.write(f'{b},{avgvol},{variation}%,{avgtime}\n')
    f.close()
    print(
        '| ' + "{:<28}".format(b) + ' | ' + "{:>8}".format(smtai_res[b]) + ' | ' + "{:>7}".format(pilat_res[b]) +
        ' || ' + "{:>7}".format(str(avgvol)) + ' | ' + "{:>13}%".format(
            str(variation)) + ' | '
        + "{:>19}".format(str(avgtime)) + ' || ' + "{:>11}/4".format(str(status)) + ' |')

print('------------------------------------------------------------------------------------------------------------------------')
