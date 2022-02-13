# -*- coding: utf-8 -*-

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from pytorch_transformers import BertPreTrainedModel,BertModel

class GraphConvolution(nn.Module):
    """
    Simple GCN layer
    """
    def __init__(self, in_features, out_features, bias=True):
        super(GraphConvolution, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.weight = nn.Parameter(torch.FloatTensor(in_features, out_features))
        if bias:
            self.bias = nn.Parameter(torch.FloatTensor(out_features))
        else:
            self.register_parameter('bias', None)

    def forward(self, text, adj):
        hidden = torch.matmul(text, self.weight)
        denom = torch.sum(adj, dim=2, keepdim=True) + 1
        output = torch.matmul(adj, hidden) / denom
        if self.bias is not None:
            return output + self.bias
        else:
            return output

class TypeGraphConvolution(nn.Module):
    """
    Simple GCN layer
    """
    def __init__(self, in_features, out_features, embedding_dim, bias=True):
        super(TypeGraphConvolution, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.weight = nn.Parameter(torch.FloatTensor(in_features, out_features))
        self.dense = nn.Linear(embedding_dim, in_features, bias=False)
        if bias:
            self.bias = nn.Parameter(torch.FloatTensor(out_features))
        else:
            self.register_parameter('bias', None)

    def forward(self, text, adj, dep_embed):
        batch_size, max_len, feat_dim = text.shape
        val_us = text.unsqueeze(dim=2)
        val_us = val_us.repeat(1, 1, max_len, 1)
        val_sum = val_us + self.dense(dep_embed)
        adj_us = adj.unsqueeze(dim=-1)
        adj_us = adj_us.repeat(1, 1, 1, feat_dim)
        hidden = torch.matmul(val_sum, self.weight)
        output = hidden.transpose(1,2) * adj_us

        output = torch.sum(output, dim=2)

        if self.bias is not None:
            return output + self.bias
        else:
            return output


class AsaTgcn(BertPreTrainedModel):
    def __init__(self, config):
        super(AsaTgcn, self).__init__(config)
        self.config = config
        self.layer_number = 3
        self.num_labels = config.num_labels
        self.num_types = config.num_types

        self.bert = BertModel(config)
        self.TGCNLayers = nn.ModuleList(([TypeGraphConvolution(config.hidden_size, config.hidden_size, config.hidden_size)
                                         for _ in range(self.layer_number)]))
        self.fc_single = nn.Linear(config.hidden_size, self.num_labels)
        self.dropout = nn.Dropout(0.1)
        self.ensemble_linear = nn.Linear(1,3)
        self.ensemble = nn.Parameter(torch.FloatTensor(3, 1))
        self.dep_embedding = nn.Embedding(self.num_types, config.hidden_size, padding_idx=0)

    def get_attention(self, val_out, dep_embed, adj):
        batch_size, max_len, feat_dim = val_out.shape
        val_us = val_out.unsqueeze(dim=2)
        val_us = val_us.repeat(1,1,max_len,1)
        val_cat = torch.cat((val_us, dep_embed), -1).float()
        atten_expand = (val_cat * val_cat.transpose(1,2))

        attention_score = torch.sum(atten_expand, dim=-1)
        attention_score = attention_score / np.power(feat_dim, 0.5)
        exp_attention_score = torch.exp(attention_score)
        exp_attention_score = torch.mul(exp_attention_score, adj.float()) # mask
        sum_attention_score = torch.sum(exp_attention_score, dim=-1).unsqueeze(dim=-1).repeat(1,1,max_len)

        attention_score = torch.div(exp_attention_score, sum_attention_score + 1e-10)
        if 'HalfTensor' in val_out.type():
            attention_score = attention_score.half()

        return attention_score

    def get_avarage(self, aspect_indices, x):
        aspect_indices_us = torch.unsqueeze(aspect_indices, 2)
        x_mask = x * aspect_indices_us
        aspect_len = (aspect_indices_us != 0).sum(dim=1)
        x_sum = x_mask.sum(dim=1)
        x_av = torch.div(x_sum, aspect_len)

        return x_av

    def forward(self, input_ids, segment_ids, valid_ids, mem_valid_ids, dep_adj_matrix, dep_value_matrix):
        sequence_output, pooled_output = self.bert(input_ids, segment_ids)
        dep_embed = self.dep_embedding(dep_value_matrix)

        batch_size, max_len, feat_dim = sequence_output.shape
        valid_output = torch.zeros(batch_size, max_len, feat_dim, device=input_ids.device).type_as(sequence_output)
        for i in range(batch_size):
            temp = sequence_output[i][valid_ids[i] == 1]
            valid_output[i][:temp.size(0)] = temp
        valid_output = self.dropout(valid_output)

        attention_score_for_output = []
        tgcn_layer_outputs = []
        seq_out = valid_output
        for tgcn in self.TGCNLayers:
            attention_score = self.get_attention(seq_out, dep_embed, dep_adj_matrix)
            attention_score_for_output.append(attention_score)
            seq_out = F.relu(tgcn(seq_out, attention_score, dep_embed))
            tgcn_layer_outputs.append(seq_out)
        tgcn_layer_outputs_pool = [self.get_avarage(mem_valid_ids, x_out) for x_out in tgcn_layer_outputs]

        x_pool = torch.stack(tgcn_layer_outputs_pool, -1)
        ensemble_out = torch.matmul(x_pool, F.softmax(self.ensemble_linear.weight, dim=0))
        ensemble_out = ensemble_out.squeeze(dim=-1)
        ensemble_out = self.dropout(ensemble_out)
        output = self.fc_single(ensemble_out)

        return output
