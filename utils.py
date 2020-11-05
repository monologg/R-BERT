import logging
import os
import random

import numpy as np
import torch
from transformers import BertTokenizer

from official_eval import official_f1

ADDITIONAL_SPECIAL_TOKENS = ["<e1>", "</e1>", "<e2>", "</e2>"]


def get_label(args):
    """
    Return the label. label.

    Args:
    """
    return [label.strip() for label in open(os.path.join(args.data_dir, args.label_file), "r", encoding="utf-8")]


def load_tokenizer(args):
    """
    Load a tokenizer.

    Args:
    """
    tokenizer = BertTokenizer.from_pretrained(args.model_name_or_path)
    tokenizer.add_special_tokens({"additional_special_tokens": ADDITIONAL_SPECIAL_TOKENS})
    return tokenizer


def write_prediction(args, output_file, preds):
    """
    For official evaluation script
    :param output_file: prediction_file_path (e.g. eval/proposed_answers.txt)
    :param preds: [0,1,0,2,18,...]
    """
    relation_labels = get_label(args)
    with open(output_file, "w", encoding="utf-8") as f:
        for idx, pred in enumerate(preds):
            f.write("{}\t{}\n".format(8001 + idx, relation_labels[pred]))


def init_logger():
    """
    Initialize the logger.

    Args:
    """
    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s -   %(message)s",
        datefmt="%m/%d/%Y %H:%M:%S",
        level=logging.INFO,
    )


def set_seed(args):
    """
    Sets random seed.

    Args:
    """
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if not args.no_cuda and torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)


def compute_metrics(preds, labels):
    """
    Compute the f1 and predictions.

    Args:
        preds: (array): write your description
        labels: (todo): write your description
    """
    assert len(preds) == len(labels)
    return acc_and_f1(preds, labels)


def simple_accuracy(preds, labels):
    """
    Calculate accuracy.

    Args:
        preds: (array): write your description
        labels: (array): write your description
    """
    return (preds == labels).mean()


def acc_and_f1(preds, labels, average="macro"):
    """
    Compute accuracy.

    Args:
        preds: (array): write your description
        labels: (list): write your description
        average: (bool): write your description
    """
    acc = simple_accuracy(preds, labels)
    return {
        "acc": acc,
        "f1": official_f1(),
    }
