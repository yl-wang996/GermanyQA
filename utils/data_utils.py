import random
import torch
import json
from pathlib import Path
import numpy as np
from utils.config import Config
class Dataset4Gquad(torch.utils.data.Dataset):
    def __init__(self, encodings):
        self.encodings = encodings

    def __getitem__(self, idx):
        return {key: torch.tensor(value[idx]) for key, value in self.encodings.items()}

    def __len__(self):
        return len(self.encodings.input_ids)

def add_token_positions(encodings, answers, tokenizer):
    start_positions = []
    end_positions = []
    for i in range(len(answers)):
        st_po = encodings.char_to_token(i, answers[i]['answer_start'])
        start_positions.append(st_po)
        ed_po = encodings.char_to_token(i, answers[i]['answer_end'] - 1)
        end_positions.append(ed_po)

        # if start position is None, the answer passage has been truncated
        if start_positions[-1] is None:
            start_positions[-1] = tokenizer.model_max_length
        if end_positions[-1] is None:
            end_positions[-1] = tokenizer.model_max_length
    encodings.update({'start_positions': start_positions, 'end_positions': end_positions})

def load_new_data():
    random.seed(Config.seed)
    train_contexts, train_questions, train_answers = load_data(Config.train_path)
    test_contexts, test_questions, test_answers = load_data(Config.test_path)
    train_len = len(train_contexts)
    test_len = len(test_contexts)
    all_context = train_contexts + test_contexts
    all_questions = train_questions + test_questions
    all_answers =  train_answers+ test_answers
    all_len = len(all_context)
    idx = [n for n in range(0,all_len)]
    random.shuffle(idx)
    all_context = np.asarray(all_context)[idx].tolist()
    all_questions = np.asarray(all_questions)[idx].tolist()
    all_answers = np.asarray(all_answers)[idx].tolist()
    new_train_contexts, new_train_questions, new_train_answers = all_context[:train_len], all_questions[:train_len], \
                                                                 all_answers[:train_len]
    new_test_contexts, new_test_questions, new_test_answers = all_context[train_len:], all_questions[train_len:], \
                                                                 all_answers[train_len:]
    return new_train_contexts,new_train_questions,new_train_answers,new_test_contexts, new_test_questions, new_test_answers

def load_data(path):
    path = Path(path)
    with open(path, 'rb') as f:
        squad_dict = json.load(f)
    contexts = []
    questions = []
    answers = []
    for group in squad_dict['data']:
        for passage in group['paragraphs']:
            context = passage['context']
            for qa in passage['qas']:
                question = qa['question']
                for answer in qa['answers']:
                    contexts.append(context)
                    questions.append(question)
                    answers.append(answer)
    return contexts, questions, answers

def add_end_idx(answers, contexts):
    for idx, (answer, context) in enumerate(zip(answers, contexts)):
        gold_text = answer['text']
        start_idx = answer['answer_start']
        end_idx = start_idx + len(gold_text)

        # sometimes squad_dataset answers are off by a character or two – fix this
        if context[start_idx:end_idx] == gold_text:
            answers[idx]['answer_end'] = end_idx
        elif context[start_idx-1:end_idx-1] == gold_text:
            answers[idx]['answer_start'] = start_idx - 1
            answers[idx]['answer_end'] = end_idx - 1     # When the gold label is off by one character
        elif context[start_idx-2:end_idx-2] == gold_text:
            answers[idx]['answer_start'] = start_idx - 2
            answers[idx]['answer_end'] = end_idx - 2     # When the gold label is off by two characters
    return answers

def random_choose(contexts,questions,answers,valp=0.2):
    random.seed(Config.seed)
    ids = [n for n in range(len(answers))]
    train_len = int(len(answers)*(1-valp))
    random.shuffle(ids)
    train_contexts = (np.asarray(contexts)[ids][:train_len]).tolist()
    train_questions = (np.asarray(questions)[ids][:train_len]).tolist()
    train_answers = (np.asarray(answers)[ids][:train_len]).tolist()
    val_contexts = (np.asarray(contexts)[ids][train_len:]).tolist()
    val_questions = (np.asarray(questions)[ids][train_len:]).tolist()
    val_answers = (np.asarray(answers)[ids][train_len:]).tolist()
    return train_contexts,train_questions,train_answers,val_contexts,val_questions,val_answers