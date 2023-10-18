# pip install -r ../../requirements.txt

exp_name="$1"


run_cmd="python train_pgm.py \
    --exp_name=$exp_name \
    --data_dir=../../datasets/adnioasis \
    --dataset=adnioasis \
    --input_res=192 \
    --parents_x sex age diagnosis \
    --input_channels=1 \
    --bs=16"

if [ "$2" = "nohup" ]
then
  nohup ${run_cmd} > $exp_name.out 2>&1 &
  echo "Started training in background with nohup, PID: $!"
else
  ${run_cmd}
fi