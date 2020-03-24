# MetaphorDetectionSharedTask

Command to train a model on metaphor detection:
python3 run_metaphor_detection.py 
--data_dir [data-folder]
--model_type roberta
--model_name_or_path roberta-large
--output_dir [output-folder]
--max_seq_length 256
--do_train
--evaluate_during_training
--do_lower_case
--per_gpu_train_batch_size 16
--per_gpu_eval_batch_size 32
--learning_rate 3e-5
--num_train_epochs 4.0
--logging_steps 25
--seed 42
--use_init_embed
--use_pos
--pos_vocab_size
--pos_dim 8

Parameters:
If you want to concanate the bottom-layer embedding with RoBERTa embedding as features, include "--use_init_embed" otherwise do not use this flag; similarly, if you want to concatenate the POS feature with RoBERTa embedding, include "--use_pos" in the command.

Note that [*] are paths you want to set, put data in the data-folder and trained model will be saved in output-folder.
During training, you'll see the training progress and the model performance on dev data.
