"""
data_utils.py
 - process metaphor data
"""

import logging
import os
import numpy as np
import pickle

logger = logging.getLogger(__name__)


class InputExample(object):
    """
    A sentence example for token classification
    """
    def __init__(self, example_id, words, pos_list, labels):
        self.example_id = example_id
        self.words = words
        self.pos_list = pos_list
        self.labels = labels


class InputFeatures(object):
    """
    Features for an example
    """
    def __init__(self, input_ids, input_mask, segment_ids, pos_ids, label_ids):
        self.input_ids = input_ids
        self.input_mask = input_mask
        self.segment_ids = segment_ids
        self.pos_ids = pos_ids
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
    """
    save_fn = data_folder+"pos_vocab.pkl"
    with open(save_fn, "wb") as handle:
        pickle.dump(handle, pos_vocab)
    print("save pos vocab to {}".format(save_fn))
    """
    

def read_examples_from_file(data_folder, mode):
    """
    @param sent_file: one sentnece per line and words are separated by space
    @param label_file: word labels separated by space, 1: metaphor, 0: non-metaphor
    """
    examples = []
    prefix = data_folder + mode + "_"
    if mode == "train":
        sent_file = open(prefix+"tokens.txt", "r")
        pos_file = open(prefix+"pos.txt", "r")
        label_file = open(prefix+"metaphor.txt", "r")
        example_id = 0
        for (sent_line, pos_line, label_line) in zip(sent_file, pos_file, label_file):
            words = sent_line.strip().split()
            pos_list = pos_line.strip().split()
            labels = [int(label) for label in label_line.strip().split()]
            examples.append(InputExample(example_id="{}-{}".format(mode, str(example_id)),
                                         words=words, pos_list=pos_list, labels=labels))
            example_id += 1
        sent_file.close()
        pos_file.close()
        label_file.close()
    # test data does not have labels
    elif mode == "test":
        sent_file = open(prefix+"tokens.txt", "r")
        pos_file = open(prefix+"pos.txt", "r")
        example_id = 0
        for (sent_line, pos_line) in zip(sent_file, pos_file):
            words = sent_line.strip().split()
            pos_list = pos_line.strip().split()
            pseudo_labels = [0] * len(words)
            examples.append(InputExample(example_id="{}-{}".format(mode, str(example_id)),
                                          words=words, pos_list=pos_list, labels=pseudo_labels))
    return examples
                                         

def convert_examples_to_features(
    examples,
    max_seq_length,
    tokenizer,
    cls_token_at_end=False,
    cls_token="[CLS]",
    cls_token_segment_id=1,
    sep_token="[SEP]",
    sep_token_extra=True,
    pad_on_left=False,
    pad_token=0,
    pad_token_segment_id=0,
    pos_vocab={},
    pad_pos_id=0,
    pad_token_label_id=-100,
    sequence_a_segment_id=0,
    mask_padding_with_zero=True):
    """
    convert examples to features as input to pretrained model
    @param cls_token_at_end: define the location of the CLS token:
        - False (Default, BERT/XLM pattern): [CLS] + A + [SEP] + B + [SEP]
        - True (XLNet/GPT pattern): A + [SEP] + B + [SEP] + [CLS]
    @param cls_token_segment_id: define the segment id associated to the CLS token
        - 0 for BERT
        - 2 for XLNet
    """
    features = []
    sent_lens = []
    for (eid, example) in enumerate(examples):
        if eid % 10000 == 0:
            logger.info("Writing example %d of %d", eid, len(examples))
        tokens = []
        pos_ids = []
        label_ids = []
        for (word, pos, label) in zip(example.words, example.pos_list, example.labels):
            # a word might be split into multiple tokens
            word_tokens = tokenizer.tokenize(word)
            tokens += list(word_tokens)
            pos_ids += [pos_vocab[pos]] + [pad_pos_id] * (len(word_tokens)-1)
            # Use the real label id for the first token of the word,
            # and padding ids for the remaining tokens
            label_ids += [label] + [-300] * (len(word_tokens)-1)
        sent_lens.append(len(tokens))
        
        # Account for [CLS] and [SEP] with "- 2" and with "- 3" for RoBERTa.
        special_tokens_count = 3 if sep_token_extra else 2
        if len(tokens) > max_seq_length - special_tokens_count:
            tokens = tokens[: (max_seq_length - special_tokens_count)]
            pos_ids = pos_ids[: (max_seq_length - special_tokens_count)]
            label_ids = label_ids[: (max_seq_length - special_tokens_count)]

        tokens += [sep_token]
        pos_ids += [pad_pos_id]
        label_ids += [pad_token_label_id]
        if sep_token_extra:
            # RoBERTa uses an extra separator b/w pairs of sentences
            tokens += [sep_token]
            pos_ids += [pad_pos_id]
            label_ids += [pad_token_label_id]
        segment_ids = [sequence_a_segment_id] * len(tokens)
            
        if cls_token_at_end:
            tokens += [cls_token]
            pos_ids += [pad_pos_id]
            label_ids += [pad_token_label_id]
            segment_ids += [cls_token_segment_id]
        else:
            tokens = [cls_token] + tokens
            pos_ids = [pad_pos_id] + pos_ids
            label_ids = [pad_token_label_id] + label_ids
            segment_ids = [cls_token_segment_id] + segment_ids

        input_ids = tokenizer.convert_tokens_to_ids(tokens)
        # The mask has 1 for real tokens and 0 for padding tokens. Only real
        # tokens are attended to.
        input_mask = [1 if mask_padding_with_zero else 0] * len(input_ids)
        # Zero-pad up to the sequence length.
        padding_length = max_seq_length - len(input_ids)
        if pad_on_left:
            input_ids = ([pad_token] * padding_length) + input_ids
            input_mask = ([0 if mask_padding_with_zero else 1] * padding_length) + input_mask
            segment_ids = ([pad_token_segment_id] * padding_length) + segment_ids
            pos_ids = ([pad_pos_id] * padding_length) + pos_ids
            label_ids = ([pad_token_label_id] * padding_length) + label_ids
        else:
            input_ids += [pad_token] * padding_length
            input_mask += [0 if mask_padding_with_zero else 1] * padding_length
            segment_ids += [pad_token_segment_id] * padding_length
            pos_ids += [pad_pos_id] * padding_length
            label_ids += [pad_token_label_id] * padding_length

        assert len(input_ids) == max_seq_length
        assert len(input_mask) == max_seq_length
        assert len(segment_ids) == max_seq_length
        assert len(pos_ids) == max_seq_length
        assert len(label_ids) == max_seq_length

        # peek data
        if eid < 4:
            logger.info("*** Example ***")
            logger.info("example_id: %s", example.example_id)
            logger.info("tokens: %s", " ".join([str(x) for x in tokens]))
            logger.info("input_ids: %s", " ".join([str(x) for x in input_ids]))
            logger.info("input_mask: %s", " ".join([str(x) for x in input_mask]))
            logger.info("segment_ids: %s", " ".join([str(x) for x in segment_ids]))
            logger.info("pos_ids: %s", " ".join([str(x) for x in pos_ids]))
            logger.info("label_ids: %s", " ".join([str(x) for x in label_ids]))
            
        features.append(InputFeatures(input_ids=input_ids,
                                      input_mask=input_mask,
                                      segment_ids=segment_ids,
                                      pos_ids=pos_ids,
                                      label_ids=label_ids))
    print("# of examples: {}, avg sent_len: {}".format(len(sent_lens), np.mean(sent_lens)))
    print("min sent len: {}, max_sent_len: {}".format(min(sent_lens), max(sent_lens)))
    return features


    

