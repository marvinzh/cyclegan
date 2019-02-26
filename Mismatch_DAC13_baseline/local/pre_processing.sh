#!/bin/bash
# Copyright 2015   David Snyder
# Apache 2.0.
#
# This script does pre-processing on extracted i-vectors:
# (1) Compute(create) files of global mean(<global_mean_and_whitening matrix-dir>/ivector.scp);
# (2) Compute(create) files of whitening matrix W(<global_mean_and_whitening matrix-dir>/transform.mat);
# (3) Do length normalization.
# Step (1) and (2) use MIXER_1phn data.

stage=0
pca_dim=400

echo "$0 $@"  # Print the command line for logging

if [ $# != 2 ]; then
  echo "Usage: $0 <dim> <sre10_1phn-ivec-dir> <global_mean_and_whitening matrix-dir>"
fi

sre_1phn_ivec_dir=$1
pre_processing_dir=$2

mkdir -p $pre_processing_dir/log

if [ $stage -le 1 ]; then
  echo "$0: Computing centering mean of SRE_1phn data"
  run.pl $pre_processing_dir/log/compute_mean.log \
  ivector-normalize-length scp:$sre_1phn_ivec_dir/ivector.scp \
  ark:- \| ivector-mean ark:- $pre_processing_dir/mean.vec || exit 1;
fi

if [ $stage -le 2 ]; then
  if [ -z "$pca_dim" ]; then
    pca_dim=-1
  fi
  echo "$0: Computing whitening transform using MIXER_1phn data"
  run.pl $pre_processing_dir/log/transform.log \
    est-pca --read-vectors=true --normalize-mean=false \
        --normalize-variance=true --dim=$pca_dim \
        scp:$sre_1phn_ivec_dir/ivector.scp $pre_processing_dir/transform.mat || exit 1;
fi
