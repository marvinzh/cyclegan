#!/bin/bash
# Copyright 2015-2017   David Snyder
#                2015   Johns Hopkins University (Author: Daniel Garcia-Romero)
#                2015   Johns Hopkins University (Author: Daniel Povey)
# Apache 2.0.
#
# 19.01.15 This script is to make domain matched system.
# See README.txt for more info on data required.
# Results (EERs) are inline in comments below.

. ./cmd.sh
. ./path.sh
set -e
mfccdir=`pwd`/mfcc
vaddir=`pwd`/mfcc
trials_female=data/sre10_test_female/trials
trials_male=data/sre10_test_male/trials
trials=data/sre10_test/trials
num_components=2048 # Larger than this doesn't make much of a difference.

date

echo "GMM-$num_components EER and mini-DCF"
for x in ind dep; do
  for y in female male pooled; do
    eer=`compute-eer <(python local/prepare_for_eer.py $trials exp/scores_gmm_${num_components}_${x}_${y}/plda_scores) 2> /dev/null`
    mindcf1=`sid/compute_min_dcf.py --p-target 0.01 exp/scores_gmm_${num_components}_${x}_${y}/plda_scores $trials 2> /dev/null`
    mindcf2=`sid/compute_min_dcf.py --p-target 0.001 exp/scores_gmm_${num_components}_${x}_${y}/plda_scores $trials 2> /dev/null`
    echo "EER ${x} ${y}: $eer"
    echo "minDCF(p-target=0.01) ${x} ${y}: $mindcf1"
    echo "minDCF(p-target=0.001) ${x} ${y}: $mindcf2"
  done
done

date
