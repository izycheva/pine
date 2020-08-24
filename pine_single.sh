# /bin/bash/
# Assign number of samples to compute volume
if (( $# < 1 )); then
	echo "Usage: ./pyinv_single.sh benchmark_name [num-volume-simulations]"
	echo "Available benchmarks:"
	echo " Non-linear:"
	echo "    pendulum_approx    rotation_nondet_small_angle"
	echo "    nonlin_example1    rotation_nondet_large_angle"
	echo "    nonlin_example2    nonlin_example3"
	echo " Linear:"
	echo "    arrow_hurwicz     symplectic"
	echo "    harmonic          pendulum_small"
	echo "    filter_goubault   filter_mine1"
	echo "    filter_mine2      filter_mine2_nondet"
	echo "    ex1               ex1_reset"
	echo "    ex2               ex2_reset"
	echo "    ex3_leadlag       ex3_reset_leadlag"
	echo "    ex4_gaussian      ex4_reset_gaussian"
	echo "    ex5_coupled_mass  ex5_reset_coupled_mass"
	echo "    ex6_butterworth   ex6_reset_butterworth"
	echo "    ex7_dampened      ex7_reset_dampened"
	echo "    ex8_harmonic      ex8_reset_harmonic"
	exit 1
else
	benchmark=$1
fi

if (( $# < 2 )); then
	volumesamples=1000000
else
	volumesamples=$2
fi


python3 -u src/exp_use_default.py $benchmark $volumesamples 0 | tee -a -i out/pyinv_$benchmark.log

