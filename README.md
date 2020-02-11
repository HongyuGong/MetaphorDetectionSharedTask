# MetaphorDetectionSharedTask

Command to train a model on metaphor detection:
python3 run_metaphor_detection.py 
--data_dir [data-folder]
--model_type roberta
--model_name_or_path roberta-large
--output_dir [output-folder]
--max_seq_length 128
--do_train
--evaluate_during_training
--do_lower_case
--per_gpu_train_batch_size 16
--per_gpu_eval_batch_size 32
--learning_rate 3e-5
--num_train_epochs 4.0
--logging_steps 25
--seed 42

Note that [*] are paths you want to set, put data in the data-folder and trained model will be saved in output-folder.
During training, you'll see the training progress and the model performance on dev data.
