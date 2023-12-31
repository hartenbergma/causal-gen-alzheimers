# train pgm model (setup=sup_pgm) and predictor/aux model (setup=sup_aux) on adnioasis dataset

# pip install -r ../../requirements.txt
# echo "Requirements installed"

pgm_name="$1"
aux_name="$2"

# python train_pgm.py \
#     --exp_name=$aux_name \
#     --data_dir=../../datasets/adnioasis/balanced_no_mci \
#     --dataset=adnioasis \
#     --input_res=192 \
#     --parents_x age diagnosis \
#     --input_channels=1 \
#     --context_norm=[-1,1] \
#     --bs=32 \
#     --lr=1e-4 \
#     --wd=0.1 \
#     --gamma=1.0 \
#     --epochs=200 \
#     --setup=sup_aux 

python train_pgm.py \
    --exp_name=$pgm_name \
    --data_dir=../../datasets/adnioasis/balanced_no_mci \
    --dataset=adnioasis \
    --input_res=192 \
    --parents_x age diagnosis \
    --input_channels=1 \
    --context_norm=[-1,1] \
    --bs=32 \
    --lr=1e-4 \
    --wd=0.1 \
    --gamma=1.0 \
    --epochs=2500 \
    --setup=sup_pgm


    # --testing=True

# python train_pgm.py \
#     --exp_name=$exp_name \
#     --data_dir=../../datasets/adnioasis/balanced \
#     --dataset=adnioasis \
#     --input_res=192 \
#     --parents_x sex age diagnosis \
#     --input_channels=1 \
#     --bs=32 \
#     --lr=5e-5 \
#     --epochs=100 \
#     --setup=sup_aux
#     # --load_path=../../checkpoints/s_a_d/aux64/checkpoint.pt \