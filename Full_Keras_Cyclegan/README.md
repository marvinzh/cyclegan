# CycleGAN_Keras

To run a new process:

1. Delete ./exp/C5/ep20/ivecxxxx/*

2. Set to_train = 1 in ./main_basic_cyc.py

3. Run ./main_basic_cyc.py for training

4. Set to_train = 0 in ./main_basic_cyc.py

5. Run ./main_basic_cyc.py to make adapted i-vectors

6. Run ./ivec_ark2scp_c5_epochs.sh

7. Copy all files under ./exp/C5/ep20/ to ../Mismatch_xxxxxx/exp/C5/ep20/

8. Run ../Mismatch_xxxxxx/run_plda.sh
