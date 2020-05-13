"""
data_utils.py
 - process metaphor data
"""

import os
import sys
import numpy as np
import pickle



class InputExample(object):
    """
    A sentence example for token classification
    """
    def __init__(self, example_id, words, pos_list, biasdown_vectors=None,
                 biasup_vectors=None, biasupdown_vectors=None,
                 corp_vectors=None, topic_vectors=None, verbnet_vectors=None,
                 wordnet_vectors=None, labels=None):
        self.example_id = example_id
        self.words = words
        self.pos_list = pos_list
        #self.content_mask_list = content_mask_list
        self.biasdown_vectors = biasdown_vectors
        self.biasup_vectors = biasup_vectors
        self.biasupdown_vectors = biasupdown_vectors
        self.corp_vectors = corp_vectors
        self.topic_vectors = topic_vectors
        self.verbnet_vectors = verbnet_vectors
        self.wordnet_vectors = wordnet_vectors
        self.labels = labels



class InputFeatures(object):
    """
    Features for an example
    """
    def __init__(self, input_ids, input_mask, segment_ids, pos_ids,
                 biasdown_vectors=None, biasup_vectors=None,
                 biasupdown_vectors=None, corp_vectors=None,
                 topic_vectors=None, verbnet_vectors=None,
                 wordnet_vectors=None, label_ids=None):
        self.input_ids = input_ids
        self.input_mask = input_mask
        self.segment_ids = segment_ids
        self.pos_ids = pos_ids
        self.biasdown_vectors = biasdown_vectors
        self.biasup_vectors = biasup_vectors
        self.biasupdown_vectors = biasupdown_vectors
        self.corp_vectors = corp_vectors
        self.topic_vectors = topic_vectors
        self.verbnet_vectors = verbnet_vectors
        self.wordnet_vectors = wordnet_vectors
        self.label_ids = label_ids


def read_pos_tags(data_folder, pos_pad="POSPAD"):
    pos_set = set()
    pos_file = open(data_folder+"train_pos.txt", "r")
    for pos_line in pos_file:
        for pos in pos_line.strip().split():
            pos_set.add(pos)
    print("# of POS tags: {}".format(len(pos_set)))
    print("POS: {}".format(pos_set))

    # the pos_pad has index 0 for <PAD> in input sequence
    pos_vocab = {0: pos_pad}
    pos_id = 1
    for pos in pos_set:
        pos_vocab[pos] = pos_id
        pos_id += 1
    return pos_vocab


def _parse_str_vector(line):
    return [[float(val) for val in str_vector.split(",")] for str_vector in line.strip().split()]


