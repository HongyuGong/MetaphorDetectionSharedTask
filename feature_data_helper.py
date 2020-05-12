"""
feature_utils.py
 - process metaphor features
"""
import argparse
import logging
import os
import sys
import numpy as np
import pickle
import jsonlines

logger = logging.getLogger(__name__)

feature2file = {"biasdown": "C-BiasDown.jsonlines", \
                "biasup": "C-BiasUp.jsonlines", \
                "biasupdown": "CCDB-BiasUpDown.jsonlines", \
                "corp": "Corpus.jsonlines", \
                "topic": "T.jsonlines", \
                "verbnet": "VN-Raw.jsonlines", \
                "wordnet": "WordNet.jsonlines"}

def readFeatureVocab(feature_fn_list):
    feature_set = set([])
    for feature_fn in feature_fn_list:
        with jsonlines.open(feature_fn) as reader:
            for obj in reader:
                feature_dict = obj["x"]
                for feature in feature_dict:
                    feature_set.add(feature)
    feature_vocab = list(feature_set)
    feature_vocab.sort()
    print("feature vocab size: {}, examples: {}".format(len(feature_vocab),
                                                        feature_vocab[:10]))
    feature2idx = {}
    for (idx, feature) in enumerate(feature_vocab):
        feature2idx[feature] = idx
    return feature_vocab, feature2idx
                

def mapTokenIdToFeature(feature_fn_list, train_type, feature_vocab_fn):
    if train_type == "train":
        # save vocabulary of features
        feature_vocab, feature2idx = readFeatureVocab(feature_fn_list)
        with open(feature_vocab_fn, "wb") as handle:
            pickle.dump((feature_vocab, feature2idx), handle)
    elif train_type == "test":
        # load vocabulary of features
        with open(feature_vocab_fn, "rb") as handle:
            feature_vocab, feature2idx = pickle.load(handle)
    feature_dim = len(feature_vocab)
    tok_id_to_feature = {}
    for feature_fn in feature_fn_list:
        with jsonlines.open(feature_fn) as reader:
            for obj in reader:
                tok_id = obj["id"]
                feature_dict = obj["x"]
                feature_vector = [0] * len(feature_vocab)
                for feature in feature_dict:
                    feature_idx = feature2idx[feature]
                    feature_vector[feature_idx] = feature_dict[feature]
                tok_id_to_feature[tok_id] = feature_vector[:]
    return tok_id_to_feature, feature_dim

        
def mapSentToTokenId(tok_id_fn):
    sent_tok_ids = []
    with open(tok_id_fn, "r") as f:
        for line in f:
            tok_ids = line.strip().split()
            sent_tok_ids.append(tok_ids[:])
    return sent_tok_ids
           

def genFeatureFile(feature_type, train_type, tok_id_fn,
                   feature_folder=None, output_folder=None):
    # read files for feature_type
    feature_affix = feature2file[feature_type]
    feature_fn_list = []
    queue = [os.path.join(feature_folder, "all_pos")]
    while queue != []:
        fn = queue.pop(0)
        if os.path.isdir(fn):
            for subfolder in os.listdir(fn):
                queue.append(os.path.join(fn, subfolder))
        elif os.path.isfile(fn) and fn.split("/")[-1] == feature_affix:
            feature_fn_list.append(fn)
        else:
            continue
    if feature_fn_list == []:
        print("No feature file for feature type: {}!".format(feature_type))
        return
    
    print("# of feature files for feature type: {}".format(feature_type))
    # map tok_id to feature vector
    feature_vocab_fn = os.path.join(output_folder, feature_type+".vocab.pkl")
    tok_id_to_feature, feature_dim = mapTokenIdToFeature(feature_fn_list,
                                                         train_type, feature_vocab_fn)

    # a list of tok_ids in a sentence
    sent_tok_ids = mapSentToTokenId(tok_id_fn)

    non_feature_cnt = 0
    # save mask in mask.txt
    mask_fout = open(os.path.join(output_folder, train_type+"_"+"mask"+".txt"), "w")
    # save feature vector in [feature_type].txt
    feature_fout = open(os.path.join(output_folder, train_type+"_"+feature_type+".txt"), "w") 
    for tok_ids in sent_tok_ids:
        tok_masks = []
        tok_features = []
        for tok_id in tok_ids:
            if tok_id in tok_id_to_feature:
                tok_masks.append(1)
                feature = tok_id_to_feature[tok_id]
                if type(feature) == type([]):
                    tok_features.append(",".join([str(val) for val in feature]))
                else:
                    #tok_features.append(str(feature))
                    print("Not a feature vector: {}".format(feature))
                    sys.exit(0)
            else:
                tok_masks.append(0)
                tok_features.append(",".join([str(0) for i in range(feature_dim)]))
                non_feature_cnt += 1
        mask_fout.write(" ".join([str(val) for val in tok_masks])+"\n")
        feature_fout.write(" ".join(tok_features)+"\n")
    mask_fout.close()
    feature_fout.close()
    print("tok # without features: {}".format(non_feature_cnt))
    print("Saving {} feature to {}".format(feature_type,
                                           os.path.join(output_folder, feature_type+".txt")))
                
    

if __name__=="__main__":
    parser = argparse.ArgumentParser()
    
    parser.add_argument(
        "--feature_dir",
        default=None,
        type=str,
        required=True,
        help="The input data dir. Should contain the training files for the CoNLL-2003 NER task.",
    )
    parser.add_argument(
        "--train_type",
        default="train",
        type=str,
        required=True,
        help="Train or test"
    )
    parser.add_argument(
        "--tok_id_fn",
        default=None,
        type=str,
        required=True,
        help="The token identifier file name"
    )
    parser.add_argument(
        "--output_dir",
        default=None,
        type=str,
        required=True,
        help="The output directory where the model predictions and checkpoints will be written.",
    )
    args = parser.parse_args()
    for feature_type in feature2file:
        genFeatureFile(feature_type, args.train_type, args.tok_id_fn,
                       args.feature_dir, args.output_dir)
        print("\n")

