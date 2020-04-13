import logging

import torch
import torch.nn as nn
from torch.nn import CrossEntropyLoss, MSELoss
import torch.nn.functional as F

from transformers.configuration_roberta import RobertaConfig
from transformers.file_utils import add_start_docstrings, add_start_docstrings_to_callable
from transformers.modeling_bert import BertEmbeddings, BertLayerNorm, BertModel, \
     BertPreTrainedModel, gelu
from transformers.modeling_roberta import RobertaEmbeddings, RobertaModel, \
     ROBERTA_PRETRAINED_MODEL_ARCHIVE_MAP, \
     ROBERTA_INPUTS_DOCSTRING

logger = logging.getLogger(__name__)

class CNNSubNetwork(nn.Module):
    def __init__(self, in_channel, num_filters, emb_size, window_sizes=(1, 32, 64), nonlin=F.leaky_relu):
        super(CNNSubNetwork, self).__init__()

        self.convs = nn.ModuleList([
            nn.Conv2d(in_channel, num_filters, (window_size, emb_size), padding = (int((window_size - 1)/2), 0))
            for window_size in window_sizes
        ])
        # self.norms = nn.ModuleList([
        #     nn.BatchNorm2d(num_filters)
        #     for window_size in window_sizes
        # ])
        self.nonlin = nonlin

    def forward(self, x):
        xs = []
        for conv_i in range(len(self.convs)):
            conv = self.convs[conv_i]
            # norm = self.norms[conv_i]
            x2 = self.nonlin(conv(x))        # [B, F, T, 1]
            x2 = torch.squeeze(x2, -1)  # [B, F, T]
            x2 = x2.permute(0, 2, 1)    # [B, T, F]
            x2 = torch.unsqueeze(x2, 1) # [B, 1, T, F]
            xs.append(x2)
            
        x = torch.cat(xs, 1)            # [B, W, T, F]
        return x

class CharCNN(nn.Module):

    def __init__(self, embedding_dim, ouput_dim, num_filters = [3, 3, 3, 3], window_sizes=(5, 7, 9), nonlin=F.leaky_relu, nonlin_dense = torch.sigmoid):
        super(CharCNN, self).__init__()
        self.conv1 = CNNSubNetwork(1, num_filters=num_filters[0], emb_size=embedding_dim, window_sizes=window_sizes, nonlin=nonlin)
        self.conv2 = CNNSubNetwork(3, num_filters=num_filters[1], emb_size=num_filters[0], window_sizes=window_sizes, nonlin=nonlin)
        self.conv3 = CNNSubNetwork(3, num_filters=num_filters[2], emb_size=num_filters[1], window_sizes=window_sizes, nonlin=nonlin)
        self.conv4 = CNNSubNetwork(3, num_filters=num_filters[3], emb_size=num_filters[3], window_sizes=window_sizes, nonlin=nonlin)
        #self.fc = nn.Linear(num_filters[3] * len(window_sizes), ouput_dim)        
        self.nonlin = nonlin
        self.nonlin_dense = nonlin_dense
        

    def forward(self, x):
        # embed = self.embedding(x) #[T, B, E]
        # embed = embed.permute(1, 0, 2) #[B, T, E]
        embed = torch.unsqueeze(x, 1) # [B, C, T, E] Add a channel dim.
        x = self.conv1(embed)
        x = self.conv2(x)
        x = self.conv3(x)
        x = self.conv4(x)
        #x = nn.MaxPool2d(kernel_size=(x.size(2), 1))(x) 
        #new = torch.zeros(x.shape[0], x.shape[1], 1, x.shape[3])
        #for i in range(len(x_len)):
        #    new[i] = torch.max(x[i, :, :x_len[i], :], dim = 1, keepdim=True)[0]
        #use x_len here 
        #x = new.view(new.size(0), -1).to(device)
        #x = self.fc(x)
        #x = self.nonlin_dense(x)
        return x

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
        self.classifier = nn.Linear(clf_dim, clf_dim)
        num_filters_char = [512, 512, 512, 512]
        self.charCNN = CharCNN(clf_dim, clf_dim, num_filters=num_filters_char)

        self.classifier2 = nn.Linear(3*num_filters_char[3], 2)

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
        outputs = self.roberta(
            input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
            position_ids=position_ids,
            head_mask=head_mask,
            inputs_embeds=inputs_embeds,
        )
        #sequence_output = torch.cat((outputs[2][24], outputs[2][23], outputs[2][22], outputs[2][21]), dim=2)
        sequence_output = outputs[2][24]
        sequence_input = outputs[2][0]
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
        hidden_output = F.leaky_relu(self.classifier(sequence_feature))
        hidden_output = self.charCNN(hidden_output)
        hidden_output = hidden_output.permute((0, 2, 1, 3))
        hidden_output = hidden_output.reshape((hidden_output.size(0), hidden_output.size(1), hidden_output.size(2)*hidden_output.size(3)))
        logits = self.classifier2(hidden_output)
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



