#!/bin/bash

. ./cmd.sh
. ./path.sh
set -e

date

echo "Create scp files for adapted ivectors."

copy-vector ark,t:exp/C5/ivectors_adpt_sre10_enroll/sre10_enroll.ark ark,scp,t:exp/C5/ivectors_adpt_sre10_enroll/ivector.ark,exp/C5/ivectors_adpt_sre10_enroll/ivector.scp
copy-vector ark,t:exp/C5/ivectors_adpt_sre10_test/sre10_test.ark ark,scp,t:exp/C5/ivectors_adpt_sre10_test/ivector.ark,exp/C5/ivectors_adpt_sre10_test/ivector.scp

echo "Task finished."

date