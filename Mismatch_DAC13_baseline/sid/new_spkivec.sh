#!/bin/bash

# Copyright     2013  Daniel Povey
#               2014  David Snyder
# Apache 2.0.

# This script extracts iVectors for a set of utterances, given
# features and a trained iVector extractor.

# Begin configuration section.
nj=30
cmd="run.pl"
stage=0
num_gselect=20 # Gaussian-selection using diagonal model: number of Gaussians to select
min_post=0.025 # Minimum posterior to use (posteriors below this are pruned out)
posterior_scale=1.0 # This scale helps to control for successve features being highly
                    # correlated.  E.g. try 0.1 or 0.3.
# End configuration section.

echo "$0 $@"  # Print the command line for logging

if [ -f path.sh ]; then . ./path.sh; fi
. parse_options.sh || exit 1;

if [ $# != 2 ]; then
  echo "Usage: $0 <data> <ivector-dir>"
  echo " e.g.: $0 data/train_male exp/ivectors_male"
  echo "main options (for others, see top of script file)"
  echo "  --config <config-file>                           # config containing options"
  echo "  --cmd (utils/run.pl|utils/queue.pl <queue opts>) # how to run jobs."
  echo "  --num-iters <#iters|10>                          # Number of iterations of E-M"
  echo "  --nj <n|10>                                      # Number of jobs (also see num-processes and num-threads)"
  echo "  --num-threads <n|8>                              # Number of threads for each process"
  echo "  --stage <stage|0>                                # To control partial reruns"
  echo "  --num-gselect <n|20>                             # Number of Gaussians to select using"
  echo "                                                   # diagonal model."
  echo "  --min-post <min-post|0.025>                      # Pruning threshold for posteriors"
  exit 1;
fi

data=$1
dir=$2

# Set various variables.
mkdir -p $dir/log
utils/split_data.sh $data $nj || exit 1;

#if [ $stage -le 2 ]; then
# Be careful here: the speaker-level iVectors are now length-normalized,
# even if they are otherwise the same as the utterance-level ones.
echo "$0: computing mean of iVectors for each speaker and length-normalizing"
$cmd $dir/log/speaker_mean.log \
  ivector-normalize-length scp:$dir/ivector.scp  ark:- \| \
  ivector-mean ark:$data/spk2utt ark:- ark:- ark,t:$dir/num_utts.ark \| \
  ivector-normalize-length ark:- ark,scp:$dir/spk_ivector.ark,$dir/spk_ivector.scp || exit 1;
#fi
