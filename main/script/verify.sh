CUDA_VISIBLE_DEVICES=0 python -u robustness_verifier.py \
    --config exp_configs/baseline_test.yaml \
    --model cifar_cnn_b \
    --model_path Train_FAT_cifar_cnn_b_seed1_2_255/model_RA_best.pth.tar \
    --output_name CNNB_baseline_cifar_2_255 \
    --start 0 --end 1000