import logging
import random
import torch
import copy
import json
from pprint import pformat
from transformers import GPT2Tokenizer, GPT2LMHeadModel
from train import add_special_tokens_, SPECIAL_TOKENS
from interact import sample_sequence
from utils import QUAC_TRAIN_URL, flatten_data, add_history, get_sentence_by_character
from datasets import load_dataset
from tqdm import tqdm

from argparse import ArgumentParser

def run():
    parser = ArgumentParser()
    parser.add_argument("--dataset_path", type=str, default=QUAC_TRAIN_URL, help="Path or url of the dataset. If empty download from S3.")
    #parser.add_argument("--dataset_cache", type=str, default='./dataset_cache', help="Path or url of the dataset cache")
    parser.add_argument("--model", type=str, default="gpt2", help="Only supports gpt2")  
    parser.add_argument("--model_checkpoint", required=True, type=str, default="", help="Path, url or short name of the model")
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu", help="Device (cuda or cpu)")
    parser.add_argument("--max_history", type=int, default="2", help="Number of dialog turns to use as history")
    parser.add_argument("--question_samples", type=int, default="3", help="Number of new questions to generate per original question")
    parser.add_argument("--output_file", type=str, default="./generated_questions.json", help="Path of file to store generated questions")

    parser.add_argument("--no_sample", action='store_true', help="Set to use greedy decoding instead of sampling")
    parser.add_argument("--max_length", type=int, default=20, help="Maximum length of the output utterances")
    parser.add_argument("--min_length", type=int, default=1, help="Minimum length of the output utterances")
    parser.add_argument("--seed", type=int, default=0, help="Seed")
    parser.add_argument("--temperature", type=float, default=0.7, help="Sampling softmax temperature")
    parser.add_argument("--top_k", type=int, default=0, help="Filter top-k tokens before sampling (<=0: no filtering)")
    parser.add_argument("--top_p", type=float, default=0.9, help="Nucleus filtering (top-p) before sampling (<=0.0: no filtering)")
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO)
    logger = logging.getLogger(__file__)
    logger.info(pformat(args))
	
    if args.seed != 0:
        random.seed(args.seed)
        torch.random.manual_seed(args.seed)
        torch.cuda.manual_seed(args.seed)

    logger.info("Get pretrained model and tokenizer")
    tokenizer_class, model_class = (GPT2Tokenizer, GPT2LMHeadModel)
    tokenizer = tokenizer_class.from_pretrained(args.model_checkpoint)
    model = model_class.from_pretrained(args.model_checkpoint)
    model.to(args.device)
    add_special_tokens_(model, tokenizer)

    if args.dataset_path == QUAC_TRAIN_URL:
        dataset = load_dataset('quac', split='train')
        dataset, turnid_idx_map = flatten_data(dataset)
        dataset = add_history(dataset, args.max_history, turnid_idx_map)

    output_dataset = []

    for turn in tqdm(dataset):
        paragraph = turn['context']
        history = turn['history']
        answer_orig_start = turn['orig_answer']['answer_start']
        paragraph_sentences = paragraph.split('. ')

        sentence_index = get_sentence_by_character(paragraph_sentences, answer_orig_start)
        low, high = (max(0, sentence_index-2), min(len(paragraph_sentences)-1, sentence_index+2))
        select_idx = random.randint(low, high)
        answer_sentence = paragraph_sentences[select_idx].strip()
        answer_sentence_segments = answer_sentence.split(', ')
        answer = answer_sentence_segments[random.randint(0, len(answer_sentence_segments)-1)]
        answer_start = paragraph.find(answer)
        
        for _ in range(args.question_samples):
            new_question = sample_sequence(paragraph, history, answer, tokenizer, model, args)
            new_question = tokenizer.decode(new_question, skip_special_tokens=True)
            turn_copy = copy.deepcopy(turn)
            turn_copy['question'] = new_question
            turn_copy['orig_answer']['text'] = answer
            turn_copy['orig_answer']['answer_start'] = answer_start
            turn_copy['orig_answer']['answer_end'] = answer_start + len(answer)
            turn_copy['answers']['texts'] = [answer]
            turn_copy['answers']['answer_starts'] = [answer_start]
            turn_copy['answer_ends'] = [answer_start + len(answer)]
            turn_copy["yesno"] = 2 # Yesno is always neither
            turn_copy["followup"] = 1 # Followup is always no

            with open(args.output_file, "a") as f:
                f.write(json.dumps(turn_copy))
                f.write('\n')

if __name__ == "__main__":
    run()