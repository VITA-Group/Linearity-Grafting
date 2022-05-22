CUDA_VISIBLE_DEVICES=0 python -u robustness_verifier.py \
    --config exp_configs/graft_test.yaml \
    --model cifar_cnn_b_graft \
    --model_path models/pretrained/cnn_b_2_255_cifar10_graft.pth \
    --output_name CNNB_graft_cifar_2_255 \
    --start 0 --end 1000