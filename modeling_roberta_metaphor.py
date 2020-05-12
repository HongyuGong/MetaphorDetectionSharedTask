import logging

import torch
import torch.nn as nn
from torch.nn import CrossEntropyLoss, MSELoss

from transformers.configuration_roberta import RobertaConfig
from transformers.file_utils import add_start_docstrings, add_start_docstrings_to_callable
from transformers.modeling_bert import BertEmbeddings, BertLayerNorm, BertModel, \
     BertPreTrainedModel, gelu
from transformers.modeling_roberta import RobertaEmbeddings, RobertaModel, \
     ROBERTA_PRETRAINED_MODEL_ARCHIVE_MAP, \
     ROBERTA_INPUTS_DOCSTRING

logger = logging.getLogger(__name__)


class RobertaForMetaphorDetection(BertPreTrainedModel):
    config_class = RobertaConfig
    pretrained_model_archive_map = ROBERTA_PRETRAINED_MODEL_ARCHIVE_MAP
    base_model_prefix = "roberta"

    def __init__(self, config, use_pos, pos_vocab_size=20, pos_dim=6,
                 use_features=False, feature_dim=128):
        super().__init__(config)
        self.num_labels = config.num_labels
        #self.use_init_embed = use_init_embed
        self.use_pos = use_pos
        self.use_features = use_features
        #self.pos_vocab_size = config.pos_vocab_size
        #self.pos_dim = config.pos_dim

        # semantic embedding from RoBERTa
        self.roberta = RobertaModel(config)
        # project roberta embedding
        #self.output_projector = nn.Linear(config.hidden_size, embed_dim)

        # project init embedding
        #if self.use_init_embed:
        #    self.input_projector = nn.Linear(config.hidden_size, embed_dim)
        
        # pos embedding
        if use_pos:
            self.pos_emb = nn.Embedding(pos_vocab_size, pos_dim)
            self.pos_emb.weight.data.uniform_(-1, 1)
        
        # dropout
        self.dropout = nn.Dropout(config.hidden_dropout_prob)
        # classifier
        logger.info("hidden_size: {}, pos_dim: {}, feature_dim: {}".format(config.hidden_size,
                                                                           pos_dim, feature_dim))
        # reduced RoBERTa embedding
        clf_dim = config.hidden_size
        # Feature: init_embed 
        #if use_init_embed:
        #    clf_dim += config.hidden_size
        # Feature: POS_embed
        if use_pos:
            clf_dim += pos_dim
        # Feature: concreteness, topic, etc.
        if use_features:
            clf_dim += feature_dim

        logger.info("classifier dim: {}".format(clf_dim))
        self.classifier = nn.Linear(clf_dim, config.num_labels)

        self.init_weights()

    @add_start_docstrings_to_callable(ROBERTA_INPUTS_DOCSTRING)
    def forward(
        self,
        input_ids=None,
        attention_mask=None,
        token_type_ids=None,
        position_ids=None,
        head_mask=None,
        inputs_embeds=None,
        labels=None,
        pos_ids=None,
        biasdown_vectors=None,
        biasup_vectors=None,
        biasupdown_vectors=None,
        corp_vectors=None,
        topic_vectors=None,
        verbnet_vectors=None,
        wordnet_vectors=None,
        class_weights=[1.0, 1.0]
    ):
        sequence_input = self.roberta.embeddings(
            input_ids=input_ids,
            token_type_ids=token_type_ids,
            position_ids=position_ids,
            inputs_embeds=inputs_embeds
        )
        #proj_sequence_input = self.input_projector(sequence_input)

        outputs = self.roberta(
            input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
            position_ids=position_ids,
            head_mask=head_mask,
            inputs_embeds=inputs_embeds,
        )
        sequence_output = outputs[0]
        #proj_sequence_output = self.output_projector(sequence_output)

        if self.use_pos:
            pos_output = self.pos_emb(pos_ids)
        # sequence_output as a feature
        sequence_feature = sequence_output
        # sequence_input as a feature
        #if self.use_init_embed:
        #    sequence_feature = torch.cat((sequence_input, sequence_feature), dim=-1)
        # POS as a feature
        if self.use_pos:
            sequence_feature = torch.cat((sequence_feature, pos_output), dim=-1)
        # External feature
        if self.use_features:
            """
            if biasdown_vectors is not None:
                sequence_feature = torch.cat((sequence_feature, biasdown_vectors), dim=-1)
            if biasup_vectors is not None:
                sequence_feature = torch.cat((sequence_feature, biasup_vectors), dim=-1)
            if biasupdown_vectors is not None:
                sequence_feature = torch.cat((sequence_feature, biasupdown_vectors), dim=-1)
            if corp_vectors is not None:
                sequence_feature = torch.cat((sequence_feature, corp_vectors), dim=-1)
            if topic_vectors is not None:
                sequence_feature = torch.cat((sequence_feature, topic_vectors), dim=-1)
            if verbnet_vectors is not None:
                sequence_feature = torch.cat((sequence_feature, verbnet_vectors), dim=-1)
            if wordnet_vectors is not None:
                sequence_feature = torch.cat((sequence_feature, wordnet_vectors), dim=-1)
            """
            external_feature = torch.cat((biasdown_vectors, biasup_vectors, biasupdown_vectors,
                                          corp_vectors, topic_vectors, verbnet_vectors, wordnet_vectors), dim=-1)
            sequence_feature = torch.cat((sequence_feature, external_feature), dim=-1)
            
        # dropout
        sequence_feature = self.dropout(sequence_feature)
        logits = self.classifier(sequence_feature)

        outputs = (logits,) + outputs[2:]  # add hidden states and attention if they are here
        if labels is not None:
            loss_fct = CrossEntropyLoss(weight=class_weights)
            # Only keep active parts of the loss
            if attention_mask is not None:
                active_loss = attention_mask.view(-1) == 1
                active_logits = logits.view(-1, self.num_labels)[active_loss]
                active_labels = labels.view(-1)[active_loss]
                loss = loss_fct(active_logits, active_labels)
            else:
                loss = loss_fct(logits.view(-1, self.num_labels), labels.view(-1))
            outputs = (loss,) + outputs

        return outputs  # (loss), scores, (hidden_states), (attentions)



