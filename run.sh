while :
do
    CUDA_VISIBLE_DEVICES=5 ./tools/dist_train.sh \
    configs/autoassign/autoassign_r50_fpn_8x2_1x_utdac.py  \
    1 
done
