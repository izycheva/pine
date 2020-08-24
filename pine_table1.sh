# /bin/bash/
# Assign number of samples to compute volume
if (( $# < 1 )); then
	volumesamples=1000000
else
	volumesamples=$1
fi


declare -a benchmarks=(
'arrow_hurwicz' \
'pendulum_approx' \
'rotation_nondet_small_angle' \
'symplectic' \
'filter_goubault' \
'filter_mine1' \
'pendulum_small' \
'harmonic' \
'rotation_nondet_large_angle' \
'filter_mine2' \
'filter_mine2_nondet' \
'nonlin_example1' \
'nonlin_example2' \
'nonlin_example3' \
'ex1' \
'ex2' \
'ex3_leadlag' \
'ex4_gaussian' \
'ex5_coupled_mass' \
'ex6_butterworth' \
'ex7_dampened' \
'ex8_harmonic' \
'ex1_reset' \
'ex2_reset' \
'ex3_reset_leadlag' \
'ex4_reset_gaussian' \
'ex5_reset_coupled_mass' \
'ex6_reset_butterworth' \
'ex7_reset_dampened' \
'ex8_reset_harmonic'
)

cmd="(python3 -u src/exp_use_default.py {1} {2}) >> out/pyinv_{1}.log"
# cmd="echo {1} {2}"
# ( {1} {2} 2>&1) | tee out/{2}.{1}-solver.log
parallel --noswap --load 100% --eta --joblog out/jobs.log $cmd ::: ${benchmarks[@]} ::: $volumesamples
parallel --noswap --load 100% --eta --joblog out/jobs.log $cmd ::: ${benchmarks[@]} ::: $volumesamples
parallel --noswap --load 100% --eta --joblog out/jobs.log $cmd ::: ${benchmarks[@]} ::: $volumesamples
parallel --noswap --load 100% --eta --joblog out/jobs.log $cmd ::: ${benchmarks[@]} ::: $volumesamples

python3 src/parse_results.py
