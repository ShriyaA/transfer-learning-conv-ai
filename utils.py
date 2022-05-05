# Copyright (c) 2019-present, HuggingFace Inc.
# All rights reserved. This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.
from datetime import datetime
import json
import logging
import os
import tarfile
import tempfile
import socket

import torch

from transformers import cached_path
from datasets import load_dataset

PERSONACHAT_URL = "https://s3.amazonaws.com/datasets.huggingface.co/personachat/personachat_self_original.json"
QUAC_TRAIN_URL = "https://s3.amazonaws.com/my89public/quac/train_v0.2.json"
QUAC_VALID_URL = "https://s3.amazonaws.com/my89public/quac/val_v0.2.json"
HF_FINETUNED_MODEL = "https://s3.amazonaws.com/models.huggingface.co/transfer-learning-chatbot/gpt_personachat_cache.tar.gz"

logger = logging.getLogger(__file__)

def download_pretrained_model():
    """ Download and extract finetuned model from S3 """
    resolved_archive_file = cached_path(HF_FINETUNED_MODEL)
    tempdir = tempfile.mkdtemp()
    logger.info("extracting archive file {} to temp dir {}".format(resolved_archive_file, tempdir))
    with tarfile.open(resolved_archive_file, 'r:gz') as archive:
        archive.extractall(tempdir)
    return tempdir


def get_dataset(tokenizer, dataset_path, dataset_cache, max_history):
    """ Get tokenized PERSONACHAT dataset from S3 or cache."""
    dataset_path = dataset_path or QUAC_TRAIN_URL
    dataset_val_path = QUAC_VALID_URL

    logger.info("Download dataset from %s", dataset_path)
    #quac_file = cached_path(dataset_path)
    
    #with open(quac_file, "r", encoding="utf-8") as f:
        #dataset = json.loads(f.read())
    
    if dataset_path == QUAC_TRAIN_URL:
        dataset = load_dataset('quac', split='train')
        dataset_val = load_dataset('quac', split='validation')
        dataset, turnid_idx_map = flatten_data(dataset)
        dataset_val, val_turnid_idx_map = flatten_data(dataset_val)
        dataset = add_history(dataset, max_history, turnid_idx_map)
        dataset_val = add_history(dataset_val, max_history, val_turnid_idx_map)
        dataset = {'train' : dataset, 'valid' : dataset_val}

    #logger.info("Tokenize and encode the dataset")
    def tokenize(obj):
        if isinstance(obj, str):
            return tokenizer.convert_tokens_to_ids(tokenizer.tokenize(obj))
        if isinstance(obj, dict):
            return dict((n, tokenize(o)) for n, o in obj.items())
        return list(tokenize(o) for o in obj)
    #dataset = tokenize(dataset)
    
    return dataset

def flatten_data(data):
    flattened_data = []
    turn_id_idx_map = {}
    single_keys = ['dialogue_id', 'wikipedia_page_title', 'background', 'section_title', 'context']
    multi_keys = ['turn_ids', 'questions', 'followups', 'yesnos']
    idx = 0
    for row in data:
        for i in range(len(row['questions'])):
            row_dict = {}
            for k in single_keys:
                row_dict[k] = row[k]
            for k in multi_keys:
                row_dict[k[:-1]] = row[k][i]
            row_dict['orig_answer'] = {'answer_start': int(row['orig_answers']['answer_starts'][i]), 'answer_end': int(row['orig_answers']['answer_starts'][i]) + len(row['orig_answers']['texts'][i]) - 1, 'text':row['orig_answers']['texts'][i]}
            row_dict['answers'] = {'answer_starts': [int(x) for x in row['answers']['answer_starts'][i]], 'texts': row['answers']['texts'][i]}
            row_dict['answer_ends'] = [x - 1 + len(row_dict['answers']['texts'][j]) for j,x in enumerate(row_dict['answers']['answer_starts'])]
            flattened_data.append(row_dict)
            turn_id_idx_map[row_dict['turn_id']] = idx
            idx += 1
    return flattened_data, turn_id_idx_map

def add_history(data, max_history, turn_id_idx_map):
    for item in data:
        item['history'] = get_history_turns(item, max_history, data, turn_id_idx_map)
    return data

def get_history_turns(item, max_history, flattened_data, turn_id_idx_map):
    turn_no = int(item['turn_id'].split('#')[-1])
    turn_history = ""
    if turn_no > 0:
        for i in range(max(turn_no-max_history, 0), turn_no):
            try:
                history_turn_id = item['turn_id'].split('#')[0] + '#' + str(i)
                history_idx = turn_id_idx_map[history_turn_id]
                turn_history += " " + flattened_data[history_idx]['question']
                turn_history += " " + flattened_data[history_idx]['orig_answer']['text']
            except KeyError:
                pass
    turn_history = turn_history.strip()
    return turn_history

class AttrDict(dict):
    def __init__(self, *args, **kwargs):
        super(AttrDict, self).__init__(*args, **kwargs)
        self.__dict__ = self


def make_logdir(model_name: str):
    """Create unique path to save results and checkpoints, e.g. runs/Sep22_19-45-59_gpu-7_gpt2"""
    # Code copied from ignite repo
    current_time = datetime.now().strftime('%b%d_%H-%M-%S')
    logdir = os.path.join(
        'runs', current_time + '_' + socket.gethostname() + '_' + model_name)
    return logdir
