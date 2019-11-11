import os
import logging
from tqdm import tqdm, trange

import numpy as np
import torch
from torch.utils.data import DataLoader, RandomSampler, SequentialSampler
from transformers import AdamW, WarmupLinearSchedule

from model import RBERT
from utils import set_seed, write_prediction, compute_metrics

logger = logging.getLogger(__name__)


class Trainer(object):
    def __init__(self, config, train_dataset=None, test_dataset=None):
        self.config = config
        self.train_dataset = train_dataset
        self.test_dataset = test_dataset

        self.model = RBERT(config)

        # GPU or CPU
        self.device = "cuda" if torch.cuda.is_available() and not config.no_cuda else "cpu"
        self.model.to(self.device)

    def train(self):
        train_sampler = RandomSampler(self.train_dataset)
        train_dataloader = DataLoader(self.train_dataset, sampler=train_sampler, batch_size=self.config.batch_size)

        if self.config.max_steps > 0:
            t_total = self.config.max_steps
            self.config.num_train_epochs = self.config.max_steps // (len(train_dataloader) // self.config.gradient_accumulation_steps) + 1
        else:
            t_total = len(train_dataloader) // self.config.gradient_accumulation_steps * self.config.num_train_epochs

        # Prepare optimizer and schedule (linear warmup and decay)
        no_decay = ['bias', 'LayerNorm.weight']
        optimizer_grouped_parameters = [
            {'params': [p for n, p in self.model.named_parameters() if not any(nd in n for nd in no_decay)],
             'weight_decay': self.config.weight_decay},
            {'params': [p for n, p in self.model.named_parameters() if any(nd in n for nd in no_decay)], 'weight_decay': 0.0}
        ]
        optimizer = AdamW(optimizer_grouped_parameters, lr=self.config.learning_rate, eps=self.config.adam_epsilon)
        scheduler = WarmupLinearSchedule(optimizer, warmup_steps=self.config.warmup_steps, t_total=t_total)

        # Train!
        logger.info("***** Running training *****")
        logger.info("  Num examples = %d", len(self.train_dataset))
        logger.info("  Num Epochs = %d", self.config.num_train_epochs)
        logger.info("  Total train batch size = %d", self.config.batch_size)
        logger.info("  Gradient Accumulation steps = %d", self.config.gradient_accumulation_steps)
        logger.info("  Total optimization steps = %d", t_total)

        global_step = 0
        tr_loss = 0.0
        self.model.zero_grad()

        train_iterator = trange(int(self.config.num_train_epochs), desc="Epoch")
        set_seed(self.config.seed)

        for _ in train_iterator:
            epoch_iterator = tqdm(train_dataloader, desc="Iteration")
            for step, batch in enumerate(epoch_iterator):
                self.model.train()
                batch = tuple(t.to(self.device) for t in batch)  # GPU or CPU
                inputs = {'input_ids': batch[0],
                          'attention_mask': batch[1],
                          'token_type_ids': batch[2],
                          'labels': batch[3],
                          'e1_mask': batch[4],
                          'e2_mask': batch[5],
                          }
                outputs = self.model(**inputs)
                loss = outputs[0]

                if self.config.gradient_accumulation_steps > 1:
                    loss = loss / self.config.gradient_accumulation_steps

                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.config.max_grad_norm)

                tr_loss += loss.item()
                if (step + 1) % self.config.gradient_accumulation_steps == 0:

                    optimizer.step()
                    scheduler.step()  # Update learning rate schedule
                    self.model.zero_grad()
                    global_step += 1

                    if self.config.save_steps > 0 and global_step % self.config.save_steps == 0:
                        # Save model checkpoint (Overwrite)
                        output_dir = os.path.join(self.config.model_dir)

                        if not os.path.exists(output_dir):
                            os.makedirs(output_dir)
                        model_to_save = self.model.module if hasattr(self.model, 'module') else self.model
                        model_to_save.save_pretrained(output_dir)
                        torch.save(self.config, os.path.join(output_dir, 'training_config.bin'))
                        logger.info("Saving model checkpoint to %s", output_dir)

                    if self.config.logging_steps > 0 and global_step % self.config.logging_steps == 0:
                        # Log metrics
                        if self.config.evaluate_during_training:  # Only evaluate when single GPU otherwise metrics may not average well
                            results = self.evaluate()

                if self.config.max_steps > 0 and global_step > self.config.max_steps:
                    epoch_iterator.close()
                    break

            if self.config.max_steps > 0 and global_step > self.config.max_steps:
                train_iterator.close()
                break

        return global_step, tr_loss / global_step

    def evaluate(self):
        # self.load_model()  # Load model

        eval_sampler = SequentialSampler(self.test_dataset)
        eval_dataloader = DataLoader(self.test_dataset, sampler=eval_sampler, batch_size=self.config.batch_size)

        # Eval!
        logger.info("***** Running evaluation *****")
        logger.info("  Num examples = %d", len(self.test_dataset))
        logger.info("  Batch size = %d", self.config.batch_size)
        eval_loss = 0.0
        nb_eval_steps = 0
        preds = None
        out_label_ids = None
        results = {}

        for batch in tqdm(eval_dataloader, desc="Evaluating"):
            self.model.eval()
            batch = tuple(t.to(self.device) for t in batch)
            with torch.no_grad():
                inputs = {'input_ids': batch[0],
                          'attention_mask': batch[1],
                          'token_type_ids': batch[2],
                          'labels': batch[3],
                          'e1_mask': batch[4],
                          'e2_mask': batch[5]}
                outputs = self.model(**inputs)
                tmp_eval_loss, logits = outputs[:2]

                eval_loss += tmp_eval_loss.mean().item()
            nb_eval_steps += 1

            if preds is None:
                preds = logits.detach().cpu().numpy()
                out_label_ids = inputs['labels'].detach().cpu().numpy()
            else:
                preds = np.append(preds, logits.detach().cpu().numpy(), axis=0)
                out_label_ids = np.append(
                    out_label_ids, inputs['labels'].detach().cpu().numpy(), axis=0)

        eval_loss = eval_loss / nb_eval_steps
        preds = np.argmax(preds, axis=1)
        result = compute_metrics(preds, out_label_ids)
        results.update(result)
        logger.info("***** Eval results *****")
        for key in sorted(result.keys()):
            logger.info("  %s = %s", key, str(result[key]))

        write_prediction(os.path.join(self.config.eval_dir, "proposed_answers.txt"), preds)
        return results

    def load_model(self):
        # Check whether model exists
        if not os.path.exists(self.config.model_dir):
            raise Exception("Model doesn't exists! Train first!")

        try:
            self.model = RBERT.from_pretrained(self.config.model_dir)
            self.model.to(self.device)
            logger.info("***** Model Loaded *****")
        except:
            raise Exception("Some model files might be missing...")
