while :
do
    CUDA_VISIBLE_DEVICES=2 ./tools/dist_train_1.sh \
    configs/deformable_detr/deformable_detr_r50_16x2_50e_dota_kld.py \
    1
done