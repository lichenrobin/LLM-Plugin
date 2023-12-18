CUDA_VISIBLE_DEVICES=0 WANDB_DISABLED=true python3 -u src/src-seqcls/run_seqcls.py \
    --model_name_or_path models/bio-linkbert-large \
    --train_file data/data-seqcls/bioasq_hf/train.json \
    --validation_file data/data-seqcls/bioasq_hf/dev.json \
    --test_file data/data-seqcls/bioasq_hf/test.json \
    --do_train --do_eval --do_predict \
    --preprocessing_num_workers 10 \
    --per_device_train_batch_size 2 \
    --gradient_accumulation_steps 16 \
    --fp16 \
    --learning_rate 3e-5 \
    --warmup_ratio 0.5 \
    --num_train_epochs 100 \
    --max_seq_length 512 \
    --save_strategy no \
    --evaluation_strategy no \
    --output_dir output/large/bioasq_train \
    --overwrite_output_dir