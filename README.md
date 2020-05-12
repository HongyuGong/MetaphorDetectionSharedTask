#IlliniMet: Illinois System for Metaphor Detection

This is IlliMet model for metaphor detection participating in [The Second Workshop on Figurative Language Processing](https://competitions.codalab.org/competitions/22188). The model takes contextualized representation from the RoBERTa model to predict metaphors. Besides, POS and external linguistic features are explored and incorporated into the model.


Requirements:

1. Python3;

2. [Transformers by Huggingface](https://github.com/huggingface/transformers).


* [VARIABLE] in the instructions below refers to the variable names that can be defined by users.

##1. Data Preparation

Obtain the datasets (VUA or TOEFL). Extract from the train data the token ids, tokens, POS tags and metaphor labels, and save them to train_ids.txt, train_tokens.txt, train_pos.txt, and train_metaphor.txt respectively. These files are in such format that each row corresponds to one sentence, and each id/token/POS/label is separated by space in a row. Save VUA data to [DATA_DIR]/VUA/, and TOEFL data to [DATA_DIR]/TOEFL/.

Process the test data in the same way as the train data, and save test_ids.txt, test_tokens.txt and test_pos.txt to the folder [DATA_DIR]/VUA/ or [DATA_DIR]/TOEFL/.


##2. Feature Preparation

###(1) Prepare VUA features

Download and unzip VUA feature files, i.e., naacl_flp_skll_train_datasets.zip and naacl_flp_skll_test_datasets.zip, from [Educational Testing Service GitHub repo](https://github.com/EducationalTestingService/metaphor/tree/master/VUA-shared-task). Adatpt features to the same format as input sentences.

Process VUA train features

```bash
python3 feature_data_helper.py
--feature_dir [DATA_DIR]/VUA/naacl_flp_skll_train_datasets/
--train_type train
--tok_id_fn [DATA_DIR]/VUA/train_ids.txt
--output_dir [DATA_DIR]/VUA/
```

Process VUA test features

```bash
python3 feature_data_helper.py
--feature_dir [DATA_DIR]/VUA/naacl_flp_skll_test_datasets/
--train_type test
--tok_id_fn [DATA_DIR]/VUA/test_ids.txt
--output_dir [DATA_DIR]/VUA/
```

###(2) Prepare TOEFL features:

Download and unzip TOEFL feature files, i.e., toefl_skll_train_features.zip and toefl_skll_test_features_no_labels.zip [Educational Testing Service GitHub repo](https://github.com/EducationalTestingService/metaphor/tree/master/TOEFL-release). The 

Process TOEFL train features:

```bash
python3 feature_data_helper.py
--feature_dir [DATA_DIR]/TOEFL/toefl_skll_train_features/
--train_type train
--tok_id_fn [DATA_DIR]/TOEFL/train_ids.txt
--output_dir [DATA_DIR]/TOEFL/
```

Process TOEFL test features:

```bash
python3 feature_data_helper.py
--feature_dir [DATA_DIR]/TOEFL/toefl_skll_test_features_no_labels/
--train_type test
--tok_id_fn [DATA_DIR]/TOEFL/test_ids.txt
--output_dir [DATA_DIR]/TOEFL/
```


##3. Train Model for Metaphor Detection

###(1) Train VUA Model

```bash
python3 run_metaphor_detection.py
--data_dir [DATA_DIR]/VUA
--model_type roberta
--model_name_or_path roberta-large
--output_dir [OUTPUT_DIR]/VUA/model/
--dataset VUA
--max_seq_length 256
--do_train
--evaluate_during_training
--do_lower_case
--per_gpu_train_batch_size 6
--per_gpu_eval_batch_size 18
--learning_rate 2e-5
--num_train_epochs 4.0
--warmup_steps 500
--seed [SEED]
--use_pos
--pos_vocab_size 43
--pos_dim 8
--use_features
--feature_dim 696
```

* DATA_DIR: the directory with data files.

* OUTPUT_DIR: the directory to save models

* SEED: a positive integer as random seed

* --use_pos: whether to use POS as input feature

* --use_features: whether to use external feature for classification


###(2) Train TOEFL Model

```bash
python3 run_metaphor_detection.py
--data_dir [DATA_DIR]/TOEFL/
--model_type roberta
--model_name_or_path roberta-large
--output_dir [OUTPUT_DIR]/TOEFL/model/
--dataset TOEFL
--max_seq_length 256
--do_train
--evaluate_during_training
--do_lower_case
--per_gpu_train_batch_size 6
--per_gpu_eval_batch_size 18
--learning_rate 2e-5
--num_train_epochs 4.0
--warmup_steps 500
--seed [SEED]
--use_pos
--pos_vocab_size 43
--pos_dim 8
--use_features
--feature_dim 215
```

* DATA_DIR: the directory with data files.

* OUTPUT_DIR: the directory to save models

* SEED: a positive integer as random seed

* --use_pos: whether to use POS as input feature

* --use_features: whether to use external feature for classification


##4. Prediction

VUA prediction from a single model

```bash
python3 run_metaphor_detection.py
--data_dir [DATA_DIR]/VUA
--model_type roberta
--model_name_or_path roberta-large
--output_dir [OUTPUT_DIR]/VUA/model/
--dataset VUA
--max_seq_length 256
--do_predict
--do_lower_case
--per_gpu_train_batch_size 6
--per_gpu_eval_batch_size 18
--learning_rate 2e-5
--num_train_epochs 4.0
--warmup_steps 500
--seed [SEED]
--use_pos
--pos_vocab_size 43
--pos_dim 8
--use_features
--feature_dim 696
```

TOEFL prediction from a single model

```bash
python3 run_metaphor_detection.py
--data_dir [DATA_DIR]/TOEFL/
--model_type roberta
--model_name_or_path roberta-large
--output_dir [OUTPUT_DIR]/TOEFL/model/
--dataset TOEFL
--max_seq_length 256
--do_train
--evaluate_during_training
--do_lower_case
--per_gpu_train_batch_size 6
--per_gpu_eval_batch_size 18
--learning_rate 2e-5
--num_train_epochs 4.0
--warmup_steps 500
--seed [SEED]
--use_pos
--pos_vocab_size 43
--pos_dim 8
--use_features
--feature_dim 215
```

##5. Emsemble Prediction

Repeat step 4 to train multiple models using different random seeds. Ensemble method is adopted to make predictions by taking majority votes among multiple trained models. Put prediction outputs from multiple models into PRED_DIR/, and the ensemble method makes final predictions in FINAL_OUTPUT_DIR/.

```bash
python3 ensemble_test.py 
--data_dir [PRED_DIR]/
--output_dir [FINAL_OUTPUT_DIR]/
```

If you have questions, please contact Hongyu Gong (hongyugong6@gmail.com).

If you use our code, please cite our work:
???



