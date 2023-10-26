# pip install -r ../../requirements.txt
# echo "Requirements installed"

exp_name="$1"
run_cmd="python train_cf.py \
    --data_dir=../../datasets/adnioasis/balanced \
    --exp_name=$exp_name \
    --pgm_path=../../checkpoints/s_a_d/pgm64/checkpoint.pt \
    --predictor_path=../../checkpoints/s_a_d/aux64/checkpoint.pt \
    --vae_path=../../checkpoints/s_a_d/vae64/checkpoint.pt \
    --hps adnioasis192 \
    --parents_x sex age diagnosis \
    --context_dim=3 \
    --concat_pa \
    --lr=1e-4 \
    --bs=32 \
    --wd=0.1 \
    --eval_freq=1 \
    --plot_freq=20 \
    --do_pa=None \
    --alpha=0.1 \
    --seed=7"

if [ "$2" = "nohup" ]
then
  nohup ${run_cmd} > $exp_name.out 2>&1 &
  echo "Started training in background with nohup, PID: $!"
else
  ${run_cmd}
fi

