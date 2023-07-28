import os
import sys
import argparse
from pathlib import Path
from torch.utils.data import RandomSampler
from transformers import BertConfig, BertForSequenceClassification, BertTokenizer

from dataset import SentimentDataset
from model import SentimentBERT

ROOT = Path().absolute()  # Current working directory
if str(ROOT) not in sys.path:
    sys.path.append(str(ROOT))  # add ROOT to PATH

BERT_MODEL = 'bert-base-uncased'
NUM_LABELS = 2  # negative and positive reviews

def parse_opt():
    parser = argparse.ArgumentParser(prog='main')
    parser.add_argument('--train', action="store_true", help="Train new weights")
    parser.add_argument('--evaluate', action="store_true",
                        help="Evaluate existing weights")
    parser.add_argument('--predict', default="", type=str,
                        help="Predict sentiment on a given sentence")
    parser.add_argument('--weights', default='weights/',
                        type=str, help="Weights path")
    parser.add_argument('--train-data', default='data/imdb_train.csv',
                        type=str, help="IMDB train file. One sentence per line.")
    parser.add_argument('--test-data', default='data/imdb_test.csv',
                        type=str, help="IMDB train file. One sentence per line.")
    return parser.parse_args()


def train(train_file, epochs=20, output_dir="weights/"):
    config = BertConfig.from_pretrained(BERT_MODEL, num_labels=NUM_LABELS)
    tokenizer = BertTokenizer.from_pretrained(BERT_MODEL, do_lower_case=True)
    model = BertForSequenceClassification.from_pretrained(
        BERT_MODEL, config=config)

    dt = SentimentDataset(tokenizer)
    dataloader = dt.prepare_dataloader(train_file, sampler=RandomSampler)
    predictor = SentimentBERT()
    predictor.train(tokenizer, dataloader, model, epochs)

    model.save_pretrained(output_dir)
    tokenizer.save_pretrained(output_dir)


def evaluate(test_file, model_dir="weights/"):
    predictor = SentimentBERT()
    predictor.load(model_dir=model_dir)

    dt = SentimentDataset(predictor.tokenizer)
    dataloader = dt.prepare_dataloader(test_file)
    score = predictor.evaluate(dataloader)
    print(score)


def predict(text, model_dir="weights/"):
    predictor = SentimentBERT()
    predictor.load(model_dir=model_dir)

    dt = SentimentDataset(predictor.tokenizer)
    dataloader = dt.prepare_dataloader_from_examples(
        [(text, -1)], sampler=None)   # text and a dummy label
    result = predictor.predict(dataloader)

    return "Positive" if result[0] == 0 else "Negative"


def main(opt):
    train_data = opt.train_data
    test_data = opt.test_data
    weights = str(opt.weights)

    if opt.train:
        os.makedirs(opt.path, exist_ok=True)
        train(train_data, epochs=10, output_dir=weights)

    if opt.evaluate:
        evaluate(test_data, model_dir=weights)

    if len(opt.predict) > 0:
        print(predict(opt.predict, model_dir=opt.path))

    print(predict("It was truly amazing experience.", model_dir=weights))



if __name__ == '__main__':
    opt = parse_opt()
    main(opt)

