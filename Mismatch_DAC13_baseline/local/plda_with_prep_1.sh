#!/bin/bash
# Copyright 2015   David Snyder
# Apache 2.0.
#
# This script trains PLDA models and does scoring.
# This script uses SRE-1phn data for pre-process (whitening and centering) on enroll/etst data.
# Each stage of this script is quite different from the original one of /sre10v1.
# Check codes below carefully if you want to use them.

use_existing_models=false
pre_processing=true
simple_length_norm=false # If true, replace the default length normalization
                         # performed in PLDA  by an alternative that
                         # normalizes the length of the iVectors to be equal
                         # to the square root of the iVector dimension.

echo "$0 $@"  # Print the command line for logging

if [ -f path.sh ]; then . ./path.sh; fi
. parse_options.sh || exit 1;

if [ $# != 10 ]; then
  echo "Usage: $0 <plda-data-dir> <enroll-data-dir> <test-data-dir> <prep_data_dir> <<prep_ivec_dir> <plda-ivec-dir> <enroll-ivec-dir> <test-ivec-dir> <trials-file> <scores-dir>"
fi

plda_data_dir=$1
enroll_data_dir=$2
test_data_dir=$3
prep_data_dir=$4
prep_ivec_dir=$5
plda_ivec_dir=$6
enroll_ivec_dir=$7
test_ivec_dir=$8
trials=$9
scores_dir=${10}

# Centering
echo "$0: Computing centering mean"
run.pl $prep_ivec_dir/log/compute_mean.log \
ivector-normalize-length scp:$prep_ivec_dir/ivector.scp \
ark:- \| ivector-mean ark:- $prep_ivec_dir/mean.vec || exit 1;

# Whitening
pca_dim=400
echo "$0: Computing whitening transform instead of LDA dim-reduction"
run.pl $prep_ivec_dir/log/transform.log \
  est-pca --read-vectors=true --normalize-mean=false \
      --normalize-variance=true --dim=$pca_dim \
      scp:$prep_ivec_dir/ivector.scp $prep_ivec_dir/transform.mat || exit 1;

# Compute i-vector means.
run.pl ${plda_ivec_dir}/log/compute_mean.log \
  ivector-normalize-length scp:${plda_ivec_dir}/ivector.scp \
  ark:- \| ivector-mean ark:- ${plda_ivec_dir}/mean.vec || exit 1;

# This script uses LDA to decrease the dimensionality prior to PLDA.
lda_dim=400
echo "$0: Computing LDA dim-reduction transform instead of whitening"
run.pl $plda_ivec_dir/log/lda.log \
  ivector-compute-lda --total-covariance-factor=0.0 --dim=$lda_dim \
  "ark:ivector-subtract-global-mean scp:${plda_ivec_dir}/ivector.scp ark:- |" \
  ark:${plda_data_dir}/utt2spk ${plda_ivec_dir}/transform.mat || exit 1;

# Train the PLDA model and do scoring
if [ "$pre_processing" == "true" ]; then
  echo "$0: Do pre process: Yes."
  run.pl $prep_ivec_dir/log/plda.log \
    ivector-compute-plda ark:$plda_data_dir/spk2utt \
    "ark:transform-vec ${plda_ivec_dir}/transform.mat scp:${plda_ivec_dir}/ivector.scp ark:- | ivector-normalize-length ark:- ark:- |" \
    $prep_ivec_dir/plda || exit 1;

  mkdir -p $scores_dir/log
  # Attention: The reason of doing one more "ivector-normalize-length" for test i-vectors than enroll i-vectors:
  # The speaker-level i-vectors (spk_ivector.scp in the script) were already length normalized by the last stage of sid/extract_ivectors.sh.
  run.pl $scores_dir/log/plda_scoring.log \
    ivector-plda-scoring --normalize-length=true \
      --num-utts=ark:${enroll_ivec_dir}/num_utts.ark \
      "ivector-copy-plda --smoothing=0.0 ${prep_ivec_dir}/plda - |" \
      "ark:ivector-mean ark:${enroll_data_dir}/spk2utt scp:${enroll_ivec_dir}/ivector.scp ark:- | ivector-subtract-global-mean ${prep_ivec_dir}/mean.vec ark:- ark:- | transform-vec ${prep_ivec_dir}/transform.mat ark:- ark:- | ivector-normalize-length ark:- ark:- |" \
      "ark:ivector-normalize-length scp:${test_ivec_dir}/ivector.scp ark:- | ivector-subtract-global-mean ${prep_ivec_dir}/mean.vec ark:- ark:- | transform-vec ${prep_ivec_dir}/transform.mat ark:- ark:- | ivector-normalize-length ark:- ark:- |" \
      "cat '$trials' | cut -d\  --fields=1,2 |" $scores_dir/plda_scores || exit 1;

else
  echo "$0: Do pre process: No."
  run.pl $plda_ivec_dir/log/plda.log \
    ivector-compute-plda ark:$plda_data_dir/spk2utt \
    "ark:transform-vec ${plda_ivec_dir}/transform.mat scp:${plda_ivec_dir}/ivector.scp ark:- | ivector-normalize-length ark:- ark:- |" \
    $plda_ivec_dir/plda || exit 1;

  mkdir -p $scores_dir/log

  run.pl $scores_dir/log/plda_scoring.log \
    ivector-plda-scoring --normalize-length=true \
      --num-utts=ark:${enroll_ivec_dir}/num_utts.ark \
      "ivector-copy-plda --smoothing=0.0 ${plda_ivec_dir}/plda - |" \
      "ark:ivector-mean ark:${enroll_data_dir}/spk2utt scp:${enroll_ivec_dir}/ivector.scp ark:- | ivector-subtract-global-mean ${plda_ivec_dir}/mean.vec ark:- ark:- | transform-vec ${plda_ivec_dir}/transform.mat ark:- ark:- | ivector-normalize-length ark:- ark:- |" \
      "ark:ivector-normalize-length scp:${test_ivec_dir}/ivector.scp ark:- | ivector-subtract-global-mean ${plda_ivec_dir}/mean.vec ark:- ark:- | transform-vec ${plda_ivec_dir}/transform.mat ark:- ark:- | ivector-normalize-length ark:- ark:- |" \
      "cat '$trials' | cut -d\  --fields=1,2 |" $scores_dir/plda_scores || exit 1;

fi
