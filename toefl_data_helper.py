"""
toefl_data_utils.py
 - process TOEFL data
 
features for TOEFL data
 - unigram
 - part-of-speech
 - topical LDA
 - concreteness
 - lemma
 - wordnet
"""

import os
import sys
import numpy as np
import pickle
from data_utils import InputExample, InputFeatures, _parse_str_vector


def read_toefl_examples_from_file(data_folder, mode):
    """
    @param sent_file: one sentnece per line and words are separated by space
    @param label_file: word labels separated by space, 1: metaphor, 0: non-metaphor
    """
    examples = []
    prefix = data_folder + mode + "_"
    if mode == "train":
        sent_file = open(prefix+"tokens.txt", "r")
        pos_file = open(prefix+"pos.txt", "r")
        #mask_file = open(prefix+"mask.txt", "r")
        biasdown_file = open(prefix+"biasdown.txt", "r")
        biasup_file = open(prefix+"biasup.txt", "r")
        biasupdown_file = open(prefix+"biasupdown.txt", "r")
        #corp_file = open(prefix+"corp.txt", "r")
        topic_file = open(prefix+"topic.txt", "r")
        #verbnet_file = open(prefix+"verbnet.txt", "r")
        wordnet_file = open(prefix+"wordnet.txt", "r")
        label_file = open(prefix+"metaphor.txt", "r")
        example_id = 0
        for (sent_line, pos_line, biasdown_line, biasup_line, biasupdown_line,
             topic_line, wordnet_line, label_line) in \
             zip(sent_file, pos_file, biasdown_file, biasup_file, biasupdown_file,
                topic_file, wordnet_file, label_file):
            words = sent_line.strip().split()
            pos_list = pos_line.strip().split()
            #content_mask_list = [int(val) for val in mask_line.strip().split()]
            biasdown_vector_list = _parse_str_vector(biasdown_line)
            biasup_vector_list = _parse_str_vector(biasup_line)
            biasupdown_vector_list = _parse_str_vector(biasupdown_line)
            #corp_vector_list = _parse_str_vector(corp_line)
            topic_vector_list = _parse_str_vector(topic_line)
            #verbnet_vector_list = _parse_str_vector(verbnet_line)
            wordnet_vector_list = _parse_str_vector(wordnet_line)
            labels = [int(label) for label in label_line.strip().split()]
            examples.append(InputExample(example_id="{}-{}".format(mode, str(example_id)),
                                         words=words, pos_list=pos_list,
                                         #content_mask_list=content_mask_list,
                                         biasdown_vectors=biasdown_vector_list,
                                         biasup_vectors=biasup_vector_list,
                                         biasupdown_vectors=biasupdown_vector_list,
                                         topic_vectors=topic_vector_list,
                                         wordnet_vectors=wordnet_vector_list,
                                         labels=labels))
            example_id += 1
        sent_file.close()
        pos_file.close()
        #mask_file.close()
        biasdown_file.close()
        biasup_file.close()
        biasupdown_file.close()
        topic_file.close()
        wordnet_file.close()
        label_file.close()
    # test data does not have labels
    elif mode == "test":
        sent_file = open(prefix+"tokens.txt", "r")
        pos_file = open(prefix+"pos.txt", "r")
        #mask_file = open(prefix+"mask.txt", "r")
        biasdown_file = open(prefix+"biasdown.txt", "r")
        biasup_file = open(prefix+"biasup.txt", "r")
        biasupdown_file = open(prefix+"biasupdown.txt", "r")
        #corp_file = open(prefix+"corp.txt", "r")
        topic_file = open(prefix+"topic.txt", "r")
        #verbnet_file = open(prefix+"verbnet.txt", "r")
        wordnet_file = open(prefix+"wordnet.txt", "r")
        example_id = 0
        for (sent_line, pos_line, biasdown_line, biasup_line, biasupdown_line,
             topic_line, wordnet_line) in \
             zip(sent_file, pos_file, biasdown_file, biasup_file, biasupdown_file,
                 topic_file, wordnet_file):
            words = sent_line.strip().split()
            pos_list = pos_line.strip().split()
            #content_mask_list = [int(val) for val in mask_line.strip().split()]
            biasdown_vector_list = _parse_str_vector(biasdown_line)
            biasup_vector_list = _parse_str_vector(biasup_line)
            biasupdown_vector_list = _parse_str_vector(biasupdown_line)
            #corp_vector_list = _parse_str_vector(corp_line)
            topic_vector_list = _parse_str_vector(topic_line)
            #verbnet_vector_list = _parse_str_vector(verbnet_line)
            wordnet_vector_list = _parse_str_vector(wordnet_line)
            pseudo_labels = [0] * len(words)
            examples.append(InputExample(example_id="{}-{}".format(mode, str(example_id)),
                                         words=words, pos_list=pos_list,
                                         #content_mask_list=content_mask_list,
                                         biasdown_vectors=biasdown_vector_list,
                                         biasup_vectors=biasup_vector_list,
                                         biasupdown_vectors=biasupdown_vector_list,
                                         topic_vectors=topic_vector_list,
                                         wordnet_vectors=wordnet_vector_list,
                                         labels=pseudo_labels))
        sent_file.close()
        pos_file.close()
        #mask_file.close()
        biasdown_file.close()
        biasup_file.close()
        biasupdown_file.close()
        #corp_file.close()
        topic_file.close()
        #verbnet_file.close()
        wordnet_file.close()
    return examples
                                         

def convert_toefl_examples_to_features(
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
    pad_feature_val=0,
    sequence_a_segment_id=0,
    mask_padding_with_zero=True,
    mode="train"):
    
    features = []
    sent_lens = []

    feature_dim_dict = {'biasdown': 17, 'biasup': 17, 'biasupdown': 66,  \
                        'topic': 100, 'wordnet': 15}
    
    for (eid, example) in enumerate(examples):
        if eid % 10000 == 0:
            print("Writing example %d of %d", eid, len(examples))
        tokens = []
        pos_ids = []
        #content_mask = []
        biasdown_vectors = []
        biasup_vectors = []
        biasupdown_vectors = []
        corp_vectors = []
        topic_vectors = []
        verbnet_vectors = []
        wordnet_vectors = []
        label_ids = []
        
        for (word, pos, biasdown_vector, biasup_vector, biasupdown_vector,
             topic_vector, wordnet_vector, label) in \
             zip(example.words, example.pos_list,
                 example.biasdown_vectors, example.biasup_vectors,
                 example.biasupdown_vectors, example.topic_vectors,
                 example.wordnet_vectors, example.labels):
            """
            # track feature dimensions
            feature_dim_dict['biasdown'] = len(biasdown_vector)
            feature_dim_dict['biasup'] = len(biasup_vector)
            feature_dim_dict['biasupdown'] = len(biasupdown_vector)
            feature_dim_dict['topic'] = len(topic_vector)
            feature_dim_dict['wordnet'] = len(wordnet_vector)
            """
            
            # a word might be split into multiple tokens
            word_tokens = tokenizer.tokenize(word)
            tokens += list(word_tokens)
            pos_ids += [pos_vocab[pos]] + [pad_pos_id] * (len(word_tokens)-1)
            biasdown_vectors += [biasdown_vector] + \
                                [[pad_feature_val]*len(biasdown_vector) for i in range(len(word_tokens)-1)]
            biasup_vectors += [biasup_vector] + \
                              [[pad_feature_val]*len(biasup_vector) for i in range(len(word_tokens)-1)]
            biasupdown_vectors += [biasupdown_vector] + \
                                 [[pad_feature_val]*len(biasupdown_vector) for i in range(len(word_tokens)-1)]
            corp_vectors += [[] for i in range(len(word_tokens))]
            topic_vectors += [topic_vector] + \
                            [[pad_feature_val]*len(topic_vector) for i in range(len(word_tokens)-1)]
            verbnet_vectors += [[] for i in range(len(word_tokens))]
            wordnet_vectors += [wordnet_vector] + \
                              [[pad_feature_val]*len(wordnet_vector) for i in range(len(word_tokens)-1)]

            # Use the real label id for the first token of the word,
            # and padding ids for the remaining tokens
            label_ids += [label] + [pad_token_label_id] * (len(word_tokens)-1)
        sent_lens.append(len(tokens))
        
        # Account for [CLS] and [SEP] with "- 2" and with "- 3" for RoBERTa.
        special_tokens_count = 3 if sep_token_extra else 2
        if len(tokens) > max_seq_length - special_tokens_count:
            tokens = tokens[: (max_seq_length - special_tokens_count)]
            pos_ids = pos_ids[: (max_seq_length - special_tokens_count)]
            biasdown_vectors = biasdown_vectors[: (max_seq_length - special_tokens_count)]
            biasup_vectors = biasup_vectors[: (max_seq_length - special_tokens_count)]
            biasupdown_vectors = biasupdown_vectors[: (max_seq_length - special_tokens_count)]
            corp_vectors = corp_vectors[: (max_seq_length - special_tokens_count)]
            topic_vectors = topic_vectors[: (max_seq_length - special_tokens_count)]
            verbnet_vectors = verbnet_vectors[: (max_seq_length - special_tokens_count)]
            wordnet_vectors = wordnet_vectors[: (max_seq_length - special_tokens_count)]
            label_ids = label_ids[: (max_seq_length - special_tokens_count)]

        tokens += [sep_token]
        pos_ids += [pad_pos_id]
        biasdown_vectors += [[pad_feature_val] * feature_dim_dict["biasdown"]]
        biasup_vectors += [[pad_feature_val] * feature_dim_dict['biasup']]
        biasupdown_vectors += [[pad_feature_val] * feature_dim_dict['biasupdown']]
        corp_vectors += [[]]
        topic_vectors += [[pad_feature_val] * feature_dim_dict['topic']]
        verbnet_vectors += [[]]
        wordnet_vectors += [[pad_feature_val] * feature_dim_dict['wordnet']]
        label_ids += [pad_token_label_id]
        if sep_token_extra:
            # RoBERTa uses an extra separator b/w pairs of sentences
            tokens += [sep_token]
            pos_ids += [pad_pos_id]
            biasdown_vectors += [[pad_feature_val] * feature_dim_dict["biasdown"]]
            biasup_vectors += [[pad_feature_val] * feature_dim_dict['biasup']]
            biasupdown_vectors += [[pad_feature_val] * feature_dim_dict['biasupdown']]
            corp_vectors += [[]]
            topic_vectors += [[pad_feature_val] * feature_dim_dict['topic']]
            verbnet_vectors += [[]]
            wordnet_vectors += [[pad_feature_val] * feature_dim_dict['wordnet']]
            label_ids += [pad_token_label_id]
        segment_ids = [sequence_a_segment_id] * len(tokens)
            
        if cls_token_at_end:
            tokens += [cls_token]
            pos_ids += [pad_pos_id]
            biasdown_vectors += [[pad_feature_val] * feature_dim_dict["biasdown"]]
            biasup_vectors += [[pad_feature_val] * feature_dim_dict['biasup']]
            biasupdown_vectors += [[pad_feature_val] * feature_dim_dict['biasupdown']]
            corp_vectors += [[]]
            topic_vectors += [[pad_feature_val] * feature_dim_dict['topic']]
            verbnet_vectors += [[]]
            wordnet_vectors += [[pad_feature_val] * feature_dim_dict['wordnet']]
            label_ids += [pad_token_label_id]
            segment_ids += [cls_token_segment_id]
        else:
            tokens = [cls_token] + tokens
            pos_ids = [pad_pos_id] + pos_ids
            biasdown_vectors = [[pad_feature_val] * feature_dim_dict["biasdown"]] + biasdown_vectors
            biasup_vectors = [[pad_feature_val] * feature_dim_dict['biasup']] + biasup_vectors
            biasupdown_vectors = [[pad_feature_val] * feature_dim_dict['biasupdown']] + biasupdown_vectors
            corp_vectors = [[]] + corp_vectors
            topic_vectors = [[pad_feature_val] * feature_dim_dict['topic']] + topic_vectors
            verbnet_vectors = [[]] + verbnet_vectors
            wordnet_vectors = [[pad_feature_val] * feature_dim_dict['wordnet']] + wordnet_vectors
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
            biasdown_vectors = [[pad_feature_val] * feature_dim_dict["biasdown"] for i in range(padding_length)] \
                               + biasdown_vectors
            biasup_vectors = [[pad_feature_val] * feature_dim_dict['biasup'] for i in range(padding_length)] \
                             + biasup_vectors
            biasupdown_vectors = [[pad_feature_val] * feature_dim_dict['biasupdown'] for i in range(padding_length)] \
                                 + biasupdown_vectors
            corp_vectors = [[] for i in range(padding_length)] + corp_vectors
            topic_vectors = [[pad_feature_val] * feature_dim_dict['topic'] for i in range(padding_length)] \
                            + topic_vectors
            verbnet_vectors = [[] for i in range(padding_length)] + verbnet_vectors
            wordnet_vectors = [[pad_feature_val] * feature_dim_dict['wordnet'] for i in range(padding_length)] \
                              + wordnet_vectors
            label_ids = ([pad_token_label_id] * padding_length) + label_ids
        else:
            input_ids += [pad_token] * padding_length
            input_mask += [0 if mask_padding_with_zero else 1] * padding_length
            segment_ids += [pad_token_segment_id] * padding_length
            pos_ids += [pad_pos_id] * padding_length
            biasdown_vectors += [[pad_feature_val] * feature_dim_dict["biasdown"] for i in range(padding_length)]
            biasup_vectors += [[pad_feature_val] * feature_dim_dict['biasup'] for i in range(padding_length)]
            biasupdown_vectors += [[pad_feature_val] * feature_dim_dict['biasupdown'] for i in range(padding_length)]
            corp_vectors += [[] for i in range(padding_length)]
            topic_vectors += [[pad_feature_val] * feature_dim_dict['topic'] for i in range(padding_length)]
            verbnet_vectors += [[] for i in range(padding_length)]
            wordnet_vectors += [[pad_feature_val] * feature_dim_dict['wordnet'] for i in range(padding_length)]
            label_ids += [pad_token_label_id] * padding_length
        
        assert len(input_ids) == max_seq_length
        assert len(input_mask) == max_seq_length
        assert len(segment_ids) == max_seq_length
        assert len(pos_ids) == max_seq_length
        assert len(biasdown_vectors) == max_seq_length
        assert len(biasup_vectors) == max_seq_length
        assert len(biasupdown_vectors) == max_seq_length
        assert len(corp_vectors) == max_seq_length
        assert len(topic_vectors) == max_seq_length
        assert len(verbnet_vectors) == max_seq_length
        assert len(wordnet_vectors) == max_seq_length
        assert len(label_ids) == max_seq_length

        # peek data
        if eid < 4:
            print("*** Example ***")
            print("example_id: %s", example.example_id)
            print("tokens: %s", " ".join([str(x) for x in tokens]))
            print("input_ids: %s", " ".join([str(x) for x in input_ids]))
            print("input_mask: %s", " ".join([str(x) for x in input_mask]))
            print("segment_ids: %s", " ".join([str(x) for x in segment_ids]))
            print("pos_ids: %s", " ".join([str(x) for x in pos_ids]))
            print("label_ids: %s", " ".join([str(x) for x in label_ids]))
            
        features.append(InputFeatures(input_ids=input_ids,
                                      input_mask=input_mask,
                                      segment_ids=segment_ids,
                                      pos_ids=pos_ids,
                                      biasdown_vectors=biasdown_vectors,
                                      biasup_vectors=biasup_vectors,
                                      biasupdown_vectors=biasupdown_vectors,
                                      corp_vectors=corp_vectors,
                                      topic_vectors=topic_vectors,
                                      verbnet_vectors=verbnet_vectors,
                                      wordnet_vectors=wordnet_vectors,
                                      label_ids=label_ids))
    print("# of examples: {}, avg sent_len: {}".format(len(sent_lens), np.mean(sent_lens)))
    print("min sent len: {}, max_sent_len: {}".format(min(sent_lens), max(sent_lens)))
    #print("feature_dim_dict: {}".format(feature_dim_dict))
    print("feature_dim: {}".format(sum([feature_dim_dict[f] for f in feature_dim_dict])))
    return features


    

