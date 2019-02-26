#!/bin/bash
# Copyright 2015   David Snyder
# Apache 2.0.
#
if [ $# != 6 ]; then
  echo "Usage: $0 <plda-data-dir> <enroll-data-dir> <test-data-dir> <plda-ivec-dir> <enroll-ivec-dir> <test-ivec-dir>"
fi
plda_data_dir=${1%/}
enroll_data_dir=${2%/}
test_data_dir=${3%/}
plda_ivec_dir=${4%/}
enroll_ivec_dir=${5%/}
test_ivec_dir=${6%/}

if [ ! -f ${test_data_dir}/trials ]; then
  echo "${test_data_dir} needs a trial file."
  exit;
fi

mkdir -p local/.tmp

# Compute gender independent and dependent i-vector means.
run.pl ${plda_ivec_dir}/log/compute_mean.log \
  ivector-normalize-length scp:${plda_ivec_dir}/ivector.scp \
  ark:- \| ivector-mean ark:- ${plda_ivec_dir}/mean.vec || exit 1;

rm -rf local/.tmp
