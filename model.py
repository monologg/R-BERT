import torch
import torch.nn as nn
from transformers import BertModel


class FCLayer(nn.Module):
    def __init__(self, input_dim, output_dim, use_activation=False):
        super(FCLayer, self).__init__()
        self.use_activation = use_activation
        self.dropout = nn.Dropout(dropout_rate)  # TODO: gin-config
        self.linear = nn.Linear(input_dim, output_dim)

    def forward(self, x):
        x = self.dropout(x)
        if self.use_activation:
            x = torch.tanh(x)
        return self.linear(x)


class RBERT(nn.Module):
    def __init__(self):
        super(RBERT, self).__init__()

        # Load pretrained bert
        self.bert = BertModel.from_pretrained("bert-base-uncased")
