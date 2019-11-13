# R-BERT

(Unofficial) Pytorch implementation of `R-BERT`: [Enriching Pre-trained Language Model with Entity Information for Relation Classification](https://arxiv.org/abs/1905.08284)

## Model Architecture

<p float="left" align="center">
    <img width="600" src="https://user-images.githubusercontent.com/28896432/68673458-1b090d00-0597-11ea-96b1-7c1453e6edbb.png" />  
</p>

### **Method**

1. **Get three vectors from BERT.**
   - [CLS] token vector
   - averaged entity_1 vector
   - averaged entity_2 vector
2. **Pass each vector to the fully-connected layers.**
   - dropout -> tanh -> fc-layer
3. **Concatenate three vectors.**
4. **Pass the concatenated vector to fully-connect layer.**
   - dropout -> fc-layer

- **_Exactly the SAME conditions_** as written in paper.
  - **Averaging** on `entity_1` and `entity_2` hidden state vectors, respectively. (including \$, # tokens)
  - **Dropout** and **Tanh** before Fully-connected layer.

## Dependencies

- python>=3.5
- torch==1.1.0
- transformers==2.1.1
- scikit-learn>=0.20.0

## How to run

```bash
$ python3 main.py --do_train --do_eval
```

- Prediction will be written on `proposed_answers.txt` in `eval` directory.

## Evaluation

```bash
$ python3 official_eval.py
# macro-averaged F1 = 88.92%
```

- Evaluate based on the official evaluation perl script.
  - MACRO-averaged f1 score (except `Other` relation)
- You can see the detailed result on `result.txt` in `eval` directory.

## References

- [Semeval 2010 Task 8 Dataset](https://drive.google.com/file/d/0B_jQiLugGTAkMDQ5ZjZiMTUtMzQ1Yy00YWNmLWJlZDYtOWY1ZDMwY2U4YjFk/view?sort=name&layout=list&num=50)
- [Semeval 2010 Task 8 Paper](https://www.aclweb.org/anthology/S10-1006.pdf)
- [NLP-progress Relation Extraction](http://nlpprogress.com/english/relationship_extraction.html)
- [https://github.com/wang-h/bert-relation-classification](https://github.com/wang-h/bert-relation-classification)
