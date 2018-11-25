echo "start training ..."

python -u train.py \
                       --data "Criteo"  --blocks 3 --heads 2  --block_shape "[64, 64, 64]" \
                       --is_save "False" --save_path "./test_code/Criteo/b3h2_64x64x64/"  \
                       --field_size 39  --run_times 1 --data_path "./" \
                       --epoch 3 --has_residual "True"  --has_wide "False" \
                       > test_code_single.out &


python -u train.py \
                       --data "Criteo" --blocks 3 --heads 2  --block_shape "[64, 64, 64]" \
                       --is_save "False" --save_path "./test_code/Criteo/b3h2_dnn_dropkeep1_400x2/"  \
                       --field_size 39  --run_times 1 --dropout_keep_prob "[1, 1, 1]" --data_path "./" \
                       --epoch 3 --has_residual "True"  --has_wide "False"  --deep_layers "[400, 400]"\
                       > ./test_code_dnn.out &
