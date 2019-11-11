import os

from sklearn.metrics import f1_score

DATA_DIR = 'data'
LABEL_FILE = 'label.txt'

RELATION_LABELS = [label.strip() for label in open(os.path.join(DATA_DIR, LABEL_FILE), 'r', encoding='utf-8')]
NUM_LABELS = len(RELATION_LABELS)

ADDITIONAL_SPECIAL_TOKENS = ["<e1>", "</e1>", "<e2>", "</e2>"]


def write_prediction(output_file, preds):
    """
    For official evaluation script
    :param output_file: prediction_file_path (e.g. eval/proposed_answers.txt)
    :param preds: [0,1,0,2,18,...]
    """
    with open(output_file, 'w', encoding='utf-8') as f:
        for idx, pred in enumerate(preds):
            f.write("{}\t{}\n".format(8001 + idx, RELATION_LABELS[pred]))
