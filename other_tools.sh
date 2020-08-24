# /bin/bash/
# Assign number of samples to compute volume
if (( $# < 1 )); then
	volumesamples=1000000
else
	volumesamples=$1
fi

declare -a tools=('smtai' 'pilat')
cmd="(python3 -u src/compute_volume.py {1} {2}) >> out/other_tools_{1}.log"
parallel --noswap --load 100% --eta --joblog out/jobs.log $cmd ::: ${tools[@]} ::: $volumesamples

python3 src/merge_csv.py
