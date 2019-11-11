import os
import random
import logging

import torch
import numpy as np
from sklearn.metrics import f1_score
from transformers.tokenization_bert import BertTokenizer

from config import Config

config = Config('config.ini')

RELATION_LABELS = [label.strip() for label in open(os.path.join(config.data_dir, config.label_file), 'r', encoding='utf-8')]
NUM_LABELS = len(RELATION_LABELS)

ADDITIONAL_SPECIAL_TOKENS = ["<e1>", "</e1>", "<e2>", "</e2>"]


def load_tokenizer(cfg):
    tokenizer = BertTokenizer.from_pretrained(cfg.pretrained_model_name)
    tokenizer.add_special_tokens({"additional_special_tokens": ADDITIONAL_SPECIAL_TOKENS})
    return tokenizer


def write_prediction(output_file, preds):
    """
    For official evaluation script
    :param output_file: prediction_file_path (e.g. eval/proposed_answers.txt)
    :param preds: [0,1,0,2,18,...]
    """
    with open(output_file, 'w', encoding='utf-8') as f:
        for idx, pred in enumerate(preds):
            f.write("{}\t{}\n".format(8001 + idx, RELATION_LABELS[pred]))


def init_logger():
    logging.basicConfig(format='%(asctime)s - %(levelname)s - %(name)s -   %(message)s',
                        datefmt='%m/%d/%Y %H:%M:%S',
                        level=logging.INFO)


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def compute_metrics(preds, labels):
    assert len(preds) == len(labels)
    return acc_and_f1(preds, labels)


def simple_accuracy(preds, labels):
    return (preds == labels).mean()


def acc_and_f1(preds, labels, average='macro'):
    acc = simple_accuracy(preds, labels)
    f1 = f1_score(y_true=labels, y_pred=preds, average='macro')
    return {
        "acc": acc,
        "f1": f1,
    }
