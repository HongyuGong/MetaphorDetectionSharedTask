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

    def __init__(self, config):
        super().__init__(config)
        self.num_labels = config.num_labels
        config.output_attentions=True
        self.roberta = RobertaModel(config)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)
        self.classifier = nn.Linear(config.hidden_size + 128, config.num_labels)

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

        outputs = self.roberta(
            input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
            position_ids=position_ids,
            head_mask=head_mask,
            inputs_embeds=inputs_embeds,
        )
        
        sequence_output = outputs[0]
        sequence_output = self.dropout(sequence_output)
        #append the pos tags, the attention weights and any other term you want to append 
        attention_output = torch.mean(outputs[2][0], dim=1)
        for i in range(1, len(outputs[2])):
            attention_output = attention_output + torch.mean(outputs[2][i], dim=1)
        #torch.div(attention_output[0][:, 4].view(len(attention_output[0][:, 4]), 1), attention_output[0]).mean()
        #sum_attention = torch.sum(attention_output, dim = 1)
        #sequence_output = torch.cat((sequence_output, sum_attention.view(sum_attention.shape[0], sum_attention.shape[1], 1)), 2)
        sequence_output = torch.cat((sequence_output, attention_output), 2)
        print(sequence_output.shape)
        logits = self.classifier(sequence_output)
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



