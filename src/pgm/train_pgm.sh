# train pgm model (setup=sup_pgm) and predictor/aux model (setup=sup_aux) on adnioasis dataset

# pip install -r ../../requirements.txt
# echo "Requirements installed"

exp_name="$1"
run_cmd="python train_pgm.py \
    --exp_name=$exp_name \
    --data_dir=../../datasets/adnioasis \
    --dataset=adnioasis \
    --input_res=192 \
    --parents_x sex age diagnosis \
    --input_channels=1 \
    --bs=32
    --epochs=300 \
    --setup=sup_aux"
    # --setup=sup_pgm
    # --load_path=../../checkpoints/s_a_d/pgm192/checkpoint.pt \
    # --testing=True

if [ "$2" = "nohup" ]
then
  nohup ${run_cmd} > $exp_name.out 2>&1 &
  echo "Started training in background with nohup, PID: $!"
else
  ${run_cmd}
fi
