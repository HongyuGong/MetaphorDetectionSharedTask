"""
ensemble_test.py

- Read predictions by each model
- Emsemble prediction using the majority vote
"""
import argparse
import os
import numpy as np

def ensemble_prediction(data_dir, res_dir, res_fn="ensemble_test_labels.txt"):
    all_pred_labels = []
    tok_fn = os.path.join(data_dir, "test_tokens.txt")
    model_index = 0
    for rel_pred_fn in os.listdir(res_dir):
        pred_fn = os.path.join(res_dir, rel_pred_fn)
        pred_file = open(pred_fn, "r")
        tok_file = open(tok_fn, "r")
        example_index = 0
        for (pred_line, tok_line) in zip(pred_file, tok_file):
            pred_label_seq = [int(val) for val in pred_line.strip().split()]
            tok_seq = tok_line.strip().split()
            while len(pred_label_seq) < len(tok_seq):
                pred_label_seq.append(0)
                print("Example {}, pred_len: {}, tok_len: {}".format(example_index,
                                                                    len(pred_label_seq), len(tok_seq)))
            if model_index == 0:
                all_pred_labels.append(np.array(pred_label_seq))
            else:
                all_pred_labels[example_index] += np.array(pred_label_seq)
            example_index += 1
        model_index += 1
        tok_file.close()
        pred_file.close()
    
    # majority vote
    ens_fn = os.path.join(res_dir, res_fn)
    with open(ens_fn, "w") as fout:
        for raw_labels in all_pred_labels:
            labels = [int(raw_label > model_index / 2.0) for raw_label in raw_labels]
            fout.write(" ".join([str(label) for label in labels])+"\n")
    print("Emsemble predictions written to {}".format(ens_fn))

    

if __name__=="__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_dir",
                        default="../data/VUA/",
                        type=str,
                        required=True,
                        help="The data folder")
    parser.add_argument("--res_dir",
                        default="../result/VUA/",
                        type=str,
                        required=True,
                        help="The parent folder of multiple output directories")
    args = parser.parse_args()

    ensemble_prediction(data_dir=args.data_dir,
                        res_dir=args.res_dir,
                        res_fn="ensemble_test_labels.txt")








