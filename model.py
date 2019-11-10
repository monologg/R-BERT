import torch
import torch.nn as nn
from transformers import BertModel


class RBERT(nn.Module):
    def __init__(self):
        super(RBERT, self).__init__()

        # Load pretrained bert
        self.bert = BertModel.from_pretrained("bert-base-uncased")
