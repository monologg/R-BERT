"""
0. Load config
1. Load dataset
2. Load Trainer
3. train
4. eval
"""
from config import Config
from trainer import Trainer
from utils import init_logger, load_tokenizer
from data_loader import load_and_cache_examples


def main():
    config = Config('config.ini')
    init_logger()
    tokenizer = load_tokenizer(config)
    train_dataset = load_and_cache_examples(config, tokenizer, evaluate=False)
    test_dataset = load_and_cache_examples(config, tokenizer, evaluate=True)
    trainer = Trainer(config, train_dataset, test_dataset)

    if config.do_train:
        trainer.train()

    if config.do_eval:
        trainer.evaluate()


if __name__ == '__main__':
    main()
