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

# Prepare the SRE 2010 evaluation data.
local/make_sre_2010_test.pl /work/lyair/sre10_te_LDC2017S06/data/eval/ data/
local/make_sre_2010_train.pl /work/lyair/sre10_te_LDC2017S06/data/eval/ data/

# Prepare SRE-1phn for computing m and W of i-vectors
#local/make_sre.pl /work/lyair/DAC13/MIXER \
#  data/sre_train

# Prepare SWB for UBM, i-vector extractor(T-matrix) and PLDA training.
local/make_swbd.pl /work/lyair/DAC13/SWB \
  data/swbd_train

#steps/make_mfcc.sh --mfcc-config conf/mfcc.conf --nj 40 --cmd "$train_cmd" \
#  data/sre_1phn_train exp/make_mfcc $mfccdir
steps/make_mfcc.sh --mfcc-config conf/mfcc.conf --nj 40 --cmd "$train_cmd" \
  data/swbd_train exp/make_mfcc $mfccdir
steps/make_mfcc.sh --mfcc-config conf/mfcc.conf --nj 40 --cmd "$train_cmd" \
  data/sre10_train exp/make_mfcc $mfccdir
steps/make_mfcc.sh --mfcc-config conf/mfcc.conf --nj 40 --cmd "$train_cmd" \
  data/sre10_test exp/make_mfcc $mfccdir

for name in swbd_train sre10_train sre10_test; do
  utils/fix_data_dir.sh data/${name}
done

#sid/compute_vad_decision.sh --nj 40 --cmd "$train_cmd" \
#  data/sre_1phn_train exp/make_vad $vaddir
sid/compute_vad_decision.sh --nj 40 --cmd "$train_cmd" \
  data/swbd_train exp/make_vad $vaddir
sid/compute_vad_decision.sh --nj 40 --cmd "$train_cmd" \
  data/sre10_train exp/make_vad $vaddir
sid/compute_vad_decision.sh --nj 40 --cmd "$train_cmd" \
  data/sre10_test exp/make_vad $vaddir

for name in swbd_train sre10_train sre10_test; do
  utils/fix_data_dir.sh data/${name}
done

date

# Reduce the amount of training data for the UBM.
#utils/subset_data_dir.sh data/train 16000 data/train_16k
#utils/subset_data_dir.sh data/train 32000 data/train_32k

# Train UBM and i-vector extractor. Use SWB data.
sid/train_diag_ubm.sh --cmd "$train_cmd --mem 20G" \
  --nj 20 --num-threads 8 \
  data/swbd_train $num_components \
  exp/diag_ubm_$num_components

date

sid/train_full_ubm.sh --nj 20 --remove-low-count-gaussians false \
  --cmd "$train_cmd --mem 20G" data/swbd_train \
  exp/diag_ubm_$num_components exp/full_ubm_$num_components

date

sid/train_ivector_extractor.sh --cmd "$train_cmd --mem 20G" \
  --ivector-dim 600 \
  --num-iters 5 exp/full_ubm_$num_components/final.ubm data/swbd_train \
  exp/extractor

date

# Extract i-vectors from MIXER, SRE10 train(development)/test data. 
sid/extract_ivectors.sh --cmd "$train_cmd --mem 8G" --nj 20 \
  exp/extractor data/sre10_train \
  exp/ivectors_sre10_train

sid/extract_ivectors.sh --cmd "$train_cmd --mem 8G" --nj 20 \
  exp/extractor data/sre10_test \
  exp/ivectors_sre10_test

date

#sid/extract_ivectors.sh --cmd "$train_cmd --mem 8G" --nj 20 \
#  exp/extractor data/mixer_train \
#  exp/ivectors_mixer_train

sid/extract_ivectors.sh --cmd "$train_cmd --mem 8G" --nj 20 \
  exp/extractor data/swbd_train \
  exp/ivectors_swbd_train

date

# Pre-processing of the extracted i-vectors (Length-Normalization)
# Use MIXER-1phn data to calculate global mean m and whitening transform W.
#local/pre_processing.sh exp/ivectors_sre_1phn_train \
#  exp/pre_processing

# Separate the i-vectors into male and female partitions and calculate
# i-vector means used by the scoring scripts.
local/scoring_common.sh data/swbd_train data/sre10_train data/sre10_test \
  exp/ivectors_swbd_train exp/ivectors_sre10_train exp/ivectors_sre10_test

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
  exp/ivectors_swbd_train exp/ivectors_sre10_train exp/ivectors_sre10_test $trials exp/scores_gmm_2048_ind_pooled
local/plda_scoring.sh --use-existing-models true data/swbd_train data/sre10_train_female data/sre10_test_female \
  exp/ivectors_swbd_train exp/ivectors_sre10_train_female exp/ivectors_sre10_test_female $trials_female exp/scores_gmm_2048_ind_female
local/plda_scoring.sh --use-existing-models true data/swbd_train data/sre10_train_male data/sre10_test_male \
  exp/ivectors_swbd_train exp/ivectors_sre10_train_male exp/ivectors_sre10_test_male $trials_male exp/scores_gmm_2048_ind_male

# Create gender dependent PLDA models and do scoring.
local/plda_scoring.sh data/swbd_train_female data/sre10_train_female data/sre10_test_female \
  exp/ivectors_swbd_train exp/ivectors_sre10_train_female exp/ivectors_sre10_test_female $trials_female exp/scores_gmm_2048_dep_female
local/plda_scoring.sh data/swbd_train_male data/sre10_train_male data/sre10_test_male \
  exp/ivectors_swbd_train exp/ivectors_sre10_train_male exp/ivectors_sre10_test_male $trials_male exp/scores_gmm_2048_dep_male

# Pool the gender dependent results.
mkdir -p exp/scores_gmm_2048_dep_pooled
cat exp/scores_gmm_2048_dep_male/plda_scores exp/scores_gmm_2048_dep_female/plda_scores \
  > exp/scores_gmm_2048_dep_pooled/plda_scores

# GMM-2048 PLDA EER
# ind pooled: 2.26
# ind female: 2.33
# ind male:   2.05
# dep female: 2.30
# dep male:   1.59
# dep pooled: 2.00
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
