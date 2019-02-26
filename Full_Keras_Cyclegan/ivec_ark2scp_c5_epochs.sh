#!/bin/bash

. ./cmd.sh
. ./path.sh
set -e

date

echo "Create scp files for adapted ivectors in different epochs."

for x in 20; do
copy-vector ark,t:exp/C5/ep${x}/ivectors_adpt_sre10_enroll/sre10_enroll.ark ark,scp,t:exp/C5/ep${x}/ivectors_adpt_sre10_enroll/ivector.ark,exp/C5/ep${x}/ivectors_adpt_sre10_enroll/ivector.scp
copy-vector ark,t:exp/C5/ep${x}/ivectors_adpt_sre10_test/sre10_test.ark ark,scp,t:exp/C5/ep${x}/ivectors_adpt_sre10_test/ivector.ark,exp/C5/ep${x}/ivectors_adpt_sre10_test/ivector.scp
copy-vector ark,t:exp/C5/ep${x}/ivectors_adpt_sre10_test_c5/sre10_test_c5.ark ark,scp,t:exp/C5/ep${x}/ivectors_adpt_sre10_test_c5/ivector.ark,exp/C5/ep${x}/ivectors_adpt_sre10_test_c5/ivector.scp

echo "Task finished in epoch ${x}."

done

date
