PYTHONPATH=. python fintune_speech_recognition.py \
    --model "damo/speech_paraformer-large_asr_nat-zh-cn-16k-common-vocab8404-pytorch" \
    --output_dir "./checkpoint" \
    --data_path "speech_asr_aishell1_trainsets" \
    --dataset_type "small" \
    --batch_bins 2000 \
    --max_epoch 50 \
    --lr 0.00005
