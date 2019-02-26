#!/bin/bash
# Copyright 2015   David Snyder
# Apache 2.0.
#
# This script trains PLDA models and does scoring.
# This script uses SRE-1phn data for pre-process (whitening and centering) on enroll/etst data.
# Each stage of this script is quite different from the original one of /sre10v1.
# Check codes below carefully if you want to use them.

use_existing_models=false
simple_length_norm=false # If true, replace the default length normalization
                         # performed in PLDA  by an alternative that
                         # normalizes the length of the iVectors to be equal
                         # to the square root of the iVector dimension.

echo "$0 $@"  # Print the command line for logging

if [ -f path.sh ]; then . ./path.sh; fi
. parse_options.sh || exit 1;

if [ $# != 8 ]; then
  echo "Usage: $0 <plda-data-dir> <enroll-data-dir> <test-data-dir> <plda-ivec-dir> <enroll-ivec-dir> <test-ivec-dir> <trials-file> <scores-dir>"
fi

plda_data_dir=$1
enroll_data_dir=$2
test_data_dir=$3
plda_ivec_dir=$4
enroll_ivec_dir=$5
test_ivec_dir=$6
trials=$7
scores_dir=$8

# This script uses LDA to decrease the dimensionality prior to PLDA.
#lda_dim=400  # Must be the same as pca_dim in the pre-processing script.
#run.pl $plda_ivec_dir/log/lda.log \
#  ivector-compute-lda --total-covariance-factor=0.0 --dim=$lda_dim \
#  "ark:ivector-subtract-global-mean scp:${plda_ivec_dir}/ivector.scp ark:- |" \
#  ark:${plda_data_dir}/utt2spk ${plda_ivec_dir}/transform.mat || exit 1;

# THE ABOVE transform.mat files from plda_ivec_dir directory
# is not used when do whitening and centering with SRE-1phn data.
# But it is always used for training PLDA model.
# Making mean.vec in plda_ivec_dir finished at scoring_common.sh script.

# Train the PLDA model
if [ "$use_existing_models" == "true" ]; then
  for f in ${plda_ivec_dir}/mean.vec ${plda_ivec_dir}/plda ; do
    [ ! -f $f ] && echo "No such file $f" && exit 1;
  done
else
  run.pl $plda_ivec_dir/log/plda.log \
    ivector-compute-plda ark:$plda_data_dir/spk2utt \
    "ark:ivector-normalize-length scp:${plda_ivec_dir}/ivector.scp  ark:- |" \
    $plda_ivec_dir/plda || exit 1;
fi

mkdir -p $scores_dir/log

# PLDA scoring
# Attention: The reason of doing one more "ivector-normalize-length" for test i-vectors than enroll i-vectors:
# The speaker-level i-vectors (spk_ivector.scp in the script) were already length normalized by the last stage of sid/extract_ivectors.sh.
run.pl $scores_dir/log/plda_scoring.log \
  ivector-plda-scoring --normalize-length=true \
    --num-utts=ark:${enroll_ivec_dir}/num_utts.ark \
    "ivector-copy-plda --smoothing=0.0 ${plda_ivec_dir}/plda - |" \
    "ark:ivector-mean ark:${enroll_data_dir}/spk2utt scp:${enroll_ivec_dir}/ivector.scp ark:- | ivector-subtract-global-mean ${plda_ivec_dir}/mean.vec scp:${enroll_ivec_dir}/spk_ivector.scp ark:- | ivector-normalize-length ark:- ark:- |" \
    "ark:ivector-normalize-length scp:${test_ivec_dir}/ivector.scp ark:- | ivector-subtract-global-mean ${plda_ivec_dir}/mean.vec ark:- ark:- | ivector-normalize-length ark:- ark:- |" \
    "cat '$trials' | cut -d\  --fields=1,2 |" $scores_dir/plda_scores || exit 1;
