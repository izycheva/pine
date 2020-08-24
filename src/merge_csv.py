import csv
from os import path

out_dir = 'out/'

if path.exists(f'{out_dir}compare_volumes_recomputed.csv'):
    print('Overriding the recomputed volumes')

if not path.exists(f'{out_dir}smtai_volumes_recomputed.csv') or not path.exists(
        f'{out_dir}pilat_volumes_recomputed.csv'):
    print(f'No files to merge. Run "./other_tools.sh" or check manually for files: {out_dir}smtai_volumes_recomputed.csv.csv and {out_dir}pilat_volumes_recomputed.csv.csv')
    exit(1)

with open(f'{out_dir}smtai_volumes_recomputed.csv', newline='') as smtai:
    with open(f'{out_dir}pilat_volumes_recomputed.csv', newline='') as pilat:
        lc = 0

        smtai_res = list(csv.reader(smtai))
        pilat_res = list(csv.reader(pilat))

        res = open(f'{out_dir}compare_volumes_recomputed.csv', 'w')
        res.write('Benchmark,SMT-AI,Pilat\n')
        for r_smtai, r_pilat in zip(smtai_res, pilat_res):
            if lc > 0:
                assert (r_smtai[0] == r_pilat[0])
                b = r_smtai[0]
                res.write(f'{b},{r_smtai[1]},{r_pilat[1]}\n')

            lc += 1
        res.close()
