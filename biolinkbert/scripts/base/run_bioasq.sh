CUDA_VISIBLE_DEVICES=0 WANDB_DISABLED=true python3 -u src/src-seqcls/run_seqcls.py \
    --model_name_or_path models/bio-linkbert-base \
    --train_file data/data-seqcls/bioasq_hf/train.json \
    --validation_file data/data-seqcls/bioasq_hf/dev.json \
    --test_file data/data-seqcls/bioasq_hf/test.json \
    --do_train --do_eval --do_predict \
    --preprocessing_num_workers 10 \
    --per_device_train_batch_size 16 \
    --gradient_accumulation_steps 1 \
    --fp16 \
    --learning_rate 2e-5 \
    --warmup_steps 100 \
    --num_train_epochs 30 \
    --max_seq_length 512 \
    --save_strategy no \
    --evaluation_strategy no \
    --output_dir output/base/bioasq_train \
    --overwrite_output_dir