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

    def __init__(self, config, use_init_embed, use_pos, pos_vocab_size=20, pos_dim=6):
        super().__init__(config)
        self.num_labels = config.num_labels
        self.use_init_embed = use_init_embed
        self.use_pos = use_pos
        #self.pos_vocab_size = config.pos_vocab_size
        #self.pos_dim = config.pos_dim

        # semantic embedding from RoBERTa
        self.roberta = RobertaModel(config)
        # pos embedding
        if use_pos:
            self.pos_emb = nn.Embedding(pos_vocab_size, pos_dim)
            self.pos_emb.weight.data.uniform_(-1, 1)
        # dropout
        self.dropout = nn.Dropout(config.hidden_dropout_prob)
        # classifier
        logger.info("hidden_size: {}, pos_dim: {}".format(config.hidden_size, pos_dim))
        # RoBERTa embedding
        clf_dim = config.hidden_size
        # Feature: init_embed 
        if use_init_embed:
            clf_dim += config.hidden_size
        # Feature: POS_embed
        if use_pos:
            clf_dim += pos_dim

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
        class_weights=[1.0, 1.0]
    ):
        r"""
        labels (:obj:`torch.LongTensor` of shape :obj:`(batch_size, sequence_length)`, `optional`, defaults to :obj:`None`):
            Labels for computing the token classification loss.
            Indices should be in ``[0, ..., config.num_labels - 1]``.
    Returns:
        :obj:`tuple(torch.FloatTensor)` comprising various elements depending on the configuration (:class:`~transformers.RobertaConfig`) and inputs:
        loss (:obj:`torch.FloatTensor` of shape :obj:`(1,)`, `optional`, returned when ``labels`` is provided) :
            Classification loss.
        scores (:obj:`torch.FloatTensor` of shape :obj:`(batch_size, sequence_length, config.num_labels)`)
            Classification scores (before SoftMax).
        hidden_states (:obj:`tuple(torch.FloatTensor)`, `optional`, returned when ``config.output_hidden_states=True``):
            Tuple of :obj:`torch.FloatTensor` (one for the output of the embeddings + one for the output of each layer)
            of shape :obj:`(batch_size, sequence_length, hidden_size)`.
            Hidden-states of the model at the output of each layer plus the initial embedding outputs.
        attentions (:obj:`tuple(torch.FloatTensor)`, `optional`, returned when ``config.output_attentions=True``):
            Tuple of :obj:`torch.FloatTensor` (one for each layer) of shape
            :obj:`(batch_size, num_heads, sequence_length, sequence_length)`.
            Attentions weights after the attention softmax, used to compute the weighted average in the self-attention
            heads.
    Examples::
        from transformers import RobertaTokenizer, RobertaForTokenClassification
        import torch
        tokenizer = RobertaTokenizer.from_pretrained('roberta-base')
        model = RobertaForTokenClassification.from_pretrained('roberta-base')
        input_ids = torch.tensor(tokenizer.encode("Hello, my dog is cute", add_special_tokens=True)).unsqueeze(0)  # Batch size 1
        labels = torch.tensor([1] * input_ids.size(1)).unsqueeze(0)  # Batch size 1
        outputs = model(input_ids, labels=labels)
        loss, scores = outputs[:2]
        """
        sequence_input = self.roberta.embeddings(
            input_ids=input_ids,
            token_type_ids=token_type_ids,
            position_ids=position_ids,
            inputs_embeds=inputs_embeds
        )

        outputs = self.roberta(
            input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
            position_ids=position_ids,
            head_mask=head_mask,
            inputs_embeds=inputs_embeds,
        )
        sequence_output = outputs[0]

        if self.use_pos:
            pos_output = self.pos_emb(pos_ids)
        # sequence_output as a feature
        sequence_feature = sequence_output
        # sequence_input as a feature
        if self.use_init_embed:
            sequence_feature = torch.cat((sequence_feature, sequence_input), dim=-1)
        # POS as a feature
        if self.use_pos:
            sequence_feature = torch.cat((sequence_feature, pos_output), dim=-1)
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



