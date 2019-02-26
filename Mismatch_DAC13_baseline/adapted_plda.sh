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
trials=data/sre10_test_c5/trials
num_components=2048 # Larger than this doesn't make much of a difference.

date

. activate tfpy36

# Remake speaker-level i-vectors files for sre10 enroll data.
for ep in ep50; do
  sid/new_spkivec.sh data/sre10_train \
    exp/C5/${ep}/ivectors_adpt_sre10_enroll
done

for ep in ep50; do
# Train a gender independent PLDA model and do scoring.
local/plda_with_prep_1.sh --pre_processing true data/swbd_train data/sre10_train data/sre10_test_c5 data/sre_1phn_train \
 exp/ivectors_sre_1phn_train exp/ivectors_swbd_train exp/C5/${ep}/ivectors_adpt_sre10_enroll exp/C5/${ep}/ivectors_adpt_sre10_test_c5 \
 $trials exp/scores_gmm_2048_ind_pooled

echo "GMM-$num_components EER and mini-DCF for C5 condition of ${ep}"
eer=`compute-eer <(python3 local/prepare_for_eer.py $trials exp/scores_gmm_${num_components}_ind_pooled/plda_scores) 2> /dev/null`
mindcf1=`sid/compute_min_dcf.py --p-target 0.01 exp/scores_gmm_2048_ind_pooled/plda_scores $trials 2> /dev/null`
mindcf2=`sid/compute_min_dcf.py --p-target 0.001 exp/scores_gmm_2048_ind_pooled/plda_scores $trials 2> /dev/null`
echo "EER: $eer"
echo "minDCF(p-target=0.01): $mindcf1"
echo "minDCF(p-target=0.001): $mindcf2"
echo

done

date
