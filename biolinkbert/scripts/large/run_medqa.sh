CUDA_VISIBLE_DEVICES=0 WANDB_DISABLED=true python3 -u src/src-mc/run_multiple_choice.py \
    --model_name_or_path models/bio-linkbert-large \
    --train_file data/data-mc/medqa_usmle_hf/train.json \
    --validation_file data/data-mc/medqa_usmle_hf/dev.json \
    --test_file data/data-mc/medqa_usmle_hf/test.json \
    --do_train --do_eval --do_predict \
    --preprocessing_num_workers 10 \
    --per_device_train_batch_size 2 \
    --gradient_accumulation_steps 32 \
    --fp16 \
    --learning_rate 3e-5 \
    --warmup_steps 500 \
    --num_train_epochs 6 \
    --max_seq_length 512 \
    --weight_decay 0.01 \
    --adam_beta1 0.9 \
    --adam_beta2 0.98 \
    --adam_epsilon 1e-6 \
    --save_strategy no \
    --evaluation_strategy no \
    --output_dir output/large/medqa_train \
    --overwrite_output_dir
