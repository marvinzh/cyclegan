#!/bin/bash
# Copyright 2015-2017   David Snyder
#                2015   Johns Hopkins University (Author: Daniel Garcia-Romero)
#                2015   Johns Hopkins University (Author: Daniel Povey)
# Apache 2.0.
#
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

. activate LY

#for ep in ep20; do
# remake "spk_ivector." files from adaptive ivectors.
sid/new_spkivec.sh --cmd "$train_cmd --mem 8G" --nj 40 \
  data/sre10_train \
  exp/C5/ivectors_adpt_sre10_enroll
#done

#sid/new_spkivec.sh --cmd "$train_cmd --mem 8G" --nj 40 \
#  data/sre10_test \
#  exp/C5/ivectors_adpt_sre10_test

#for ep in ep20; do
# Separate the i-vectors into male and female partitions and calculate
# i-vector means used by the scoring scripts.
local/scoring_common_1.sh data/swbd_train data/sre10_train data/sre10_test \
  exp/ivectors_swbd_train exp/C5/ivectors_adpt_sre10_enroll exp/C5/ivectors_adpt_sre10_test

# The commented out scripts show how to do cosine scoring with and without
# first reducing the i-vector dimensionality with LDA. PLDA tends to work
# best, so we don't focus on the scores obtained here.
#
# local/cosine_scoring.sh data/sre10_train data/sre10_test \
#  exp/ivectors_sre10_train exp/ivectors_sre10_test $trials exp/scores_gmm_2048_ind_pooled
# local/lda_scoring.sh data/sre data/sre10_train data/sre10_test \
#  exp/ivectors_sre exp/ivectors_sre10_train exp/ivectors_sre10_test $trials exp/scores_gmm_2048_ind_pooled

# Create a gender independent PLDA model and do scoring.
local/plda_scoring.sh data/swbd_train data/sre10_train data/sre10_test \
  exp/ivectors_swbd_train exp/C5/${ep}/ivectors_adpt_sre10_enroll exp/C5/${ep}/ivectors_adpt_sre10_test $trials exp/scores_gmm_2048_ind_pooled
#done

#local/plda_scoring.sh --use-existing-models true data/swbd_train data/sre10_train_female data/sre10_test_female \
#  exp/ivectors_swbd_train exp/ivectors_sre10_train_female exp/ivectors_sre10_test_female $trials_female exp/scores_gmm_2048_ind_female

#local/plda_scoring.sh --use-existing-models true data/swbd_train data/sre10_train_male data/sre10_test_male \
#  exp/ivectors_swbd_train exp/ivectors_sre10_train_male exp/ivectors_sre10_test_male $trials_male exp/scores_gmm_2048_ind_male

# Create gender dependent PLDA models and do scoring.
#local/plda_scoring.sh data/swbd_train_female data/sre10_train_female data/sre10_test_female \
#  exp/ivectors_swbd_train exp/ivectors_sre10_train_female exp/ivectors_sre10_test_female $trials_female exp/scores_gmm_2048_dep_female
#local/plda_scoring.sh data/swbd_train_male data/sre10_train_male data/sre10_test_male \
#  exp/ivectors_swbd_train exp/ivectors_sre10_train_male exp/ivectors_sre10_test_male $trials_male exp/scores_gmm_2048_dep_male

# Pool the gender dependent results.
#mkdir -p exp/scores_gmm_2048_dep_pooled
#cat exp/scores_gmm_2048_dep_male/plda_scores exp/scores_gmm_2048_dep_female/plda_scores \
#  > exp/scores_gmm_2048_dep_pooled/plda_scores

echo "GMM-$num_components EER and mini-DCF for C5 ext condition"
eer=`compute-eer <(python3 local/prepare_for_eer.py $trials exp/scores_gmm_${num_components}_ind_pooled/plda_scores) 2> /dev/null`
mindcf1=`sid/compute_min_dcf.py --p-target 0.01 exp/scores_gmm_2048_ind_pooled/plda_scores $trials 2> /dev/null`
mindcf2=`sid/compute_min_dcf.py --p-target 0.001 exp/scores_gmm_2048_ind_pooled/plda_scores $trials 2> /dev/null`
echo "EER: $eer"
echo "minDCF(p-target=0.01): $mindcf1"
echo "minDCF(p-target=0.001): $mindcf2"

date
