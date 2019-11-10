from sklearn.metrics import f1_score

# metrics
# tqdm


RELATION_LABELS = ['Other',
                   'Cause-Effect(e1,e2)', 'Cause-Effect(e2,e1)',
                   'Instrument-Agency(e1,e2)', 'Instrument-Agency(e2,e1)',
                   'Product-Producer(e1,e2)', 'Product-Producer(e2,e1)',
                   'Content-Container(e1,e2)', 'Content-Container(e2,e1)',
                   'Entity-Origin(e1,e2)', 'Entity-Origin(e2,e1)',
                   'Entity-Destination(e1,e2)', 'Entity-Destination(e2,e1)',
                   'Component-Whole(e1,e2)', 'Component-Whole(e2,e1)',
                   'Member-Collection(e1,e2)', 'Member-Collection(e2,e1)',
                   'Message-Topic(e1,e2)', 'Message-Topic(e2,e1)']


def write_prediction(output_file, preds):
    """
    output_file: eval/proposed_answers.txt
    preds: [0,1,0,2,18,...]
    """
    with open(output_file, 'w', encoding='utf-8') as f:
        for idx, pred in enumerate(preds):
            f.write("{}\t{}\n".format(8001+idx, RELATION_LABELS[pred]))
