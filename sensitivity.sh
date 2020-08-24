# /bin/bash/
# Comment out benchmarks you want to exclude from the sensitivity experiment
declare -a benchmarks=(
     'rotation_nondet_small_angle.txt' \
     'symplectic.txt' \
     'pendulum_approx.txt' \
     'filter_goubault.txt' \
     'filter_mine1.txt' \
     'pendulum_small.txt'
     'harmonic.txt' \
     'rotation_nondet_large_angle.txt' \
     'filter_mine2.txt' \
     'filter_mine2_nondet.txt'\
     'nonlin_example1.txt' \
     'nonlin_example2.txt' \
     'nonlin_example3.txt'
     )

# Assign number of samples to compute volume
if (( $# < 1 )); then
	volumesamples=1000000
else
	volumesamples=$1
fi

declare -a cexs=('0' '1' '2' '5')
declare -a cexiters=('100' '500' '0')
declare -a samples=('100_1000' '1000_1000' '100_10000')
declare -a dist=('0.1' '0.25' '0.5')
declare -a symm=('0_0' '0_1' '1_0' '1_1')
#declare -a nb=('0' '1')
cmd="(python3 -u src/parallel_eval_all.py {1} {2} {3} {4} {5} {6}) >> out/sensitivity/parallel_log_{2}_{1}.log"
# ( {1} {2} 2>&1) | tee out/{2}.{1}-solver.log
parallel --noswap --load 100% --eta --joblog out/jobs.log $cmd ::: ${benchmarks[@]} ::: ${samples[@]} ::: ${cexs[@]} ::: ${dist[@]} ::: ${cexiters[@]} ::: ${symm[@]}

python3 src/parameter_exploration.py $volumesamples