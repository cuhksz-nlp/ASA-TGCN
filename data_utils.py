import os
import copy
import numpy as np
import torch
from torch.utils.data import Dataset
from pytorch_transformers import BertTokenizer


def pad_and_truncate(sequence, maxlen, dtype='int64', padding='post', truncating='post', value=0):
    x = (np.ones(maxlen) * value).astype(dtype)
    if truncating == 'pre':
        trunc = sequence[-maxlen:]
    else:
        trunc = sequence[:maxlen]
    trunc = np.asarray(trunc, dtype=dtype)
    if padding == 'post':
        x[:len(trunc)] = trunc
    else:
        x[-len(trunc):] = trunc
    return x


class Tokenizer4Bert:
    def __init__(self, max_seq_len, pretrained_bert_name):
        self.tokenizer = BertTokenizer.from_pretrained(pretrained_bert_name)
        self.max_seq_len = max_seq_len

    def text_to_sequence(self, text, reverse=False, padding='post', truncating='post'):
        sequence = self.tokenizer.convert_tokens_to_ids(self.tokenizer.tokenize(text))
        if len(sequence) == 0:
            sequence = [0]
        if reverse:
            sequence = sequence[::-1]
        return pad_and_truncate(sequence, self.max_seq_len, padding=padding, truncating=truncating)

    def id_to_sequence(self, sequence, reverse=False, padding='post', truncating='post'):
        if len(sequence) == 0:
            sequence = [0]
        if reverse:
            sequence = sequence[::-1]
        return pad_and_truncate(sequence, self.max_seq_len, padding=padding, truncating=truncating)

class DepInstanceParser():
    def __init__(self, basicDependencies, tokens):
        self.basicDependencies = basicDependencies
        self.tokens = tokens
        self.words = []
        self.dep_governed_info = []
        self.dep_parsing()


    def dep_parsing(self):
        if len(self.tokens) > 0:
            words = []
            for token in self.tokens:
                token['word'] = token
                words.append(self.change_word(token['word']))
            dep_governed_info = [
                {"word": word}
                for i,word in enumerate(words)
            ]
            self.words = words
        else:
            dep_governed_info = [{}] * len(self.basicDependencies)
        for dep in self.basicDependencies:
            dependent_index = dep['dependent'] - 1
            governed_index = dep['governor'] - 1
            dep_governed_info[dependent_index] = {
                "governor": governed_index,
                "dep": dep['dep']
            }
        self.dep_governed_info = dep_governed_info

    def change_word(self, word):
        if "-RRB-" in word:
            return word.replace("-RRB-", ")")
        if "-LRB-" in word:
            return word.replace("-LRB-", "(")
        return word

    def get_first_order(self, direct=False):
        dep_adj_matrix  = [[0] * len(self.dep_governed_info) for _ in range(len(self.dep_governed_info))]
        dep_type_matrix = [["none"] * len(self.dep_governed_info) for _ in range(len(self.dep_governed_info))]
        # for i in range(len(self.dep_governed_info)):
        #     dep_adj_matrix[i][i]  = 1
        #     dep_type_matrix[i][i] = "self_loop"
        for i, dep_info in enumerate(self.dep_governed_info):
            governor = dep_info["governor"]
            dep_type = dep_info["dep"]
            dep_adj_matrix[i][governor] = 1
            dep_adj_matrix[governor][i] = 1
            dep_type_matrix[i][governor] = dep_type if direct is False else "{}_in".format(dep_type)
            dep_type_matrix[governor][i] = dep_type if direct is False else "{}_out".format(dep_type)
        return dep_adj_matrix, dep_type_matrix

    def get_next_order(self, dep_adj_matrix, dep_type_matrix):
        new_dep_adj_matrix = copy.deepcopy(dep_adj_matrix)
        new_dep_type_matrix = copy.deepcopy(dep_type_matrix)
        for target_index in range(len(dep_adj_matrix)):
            for first_order_index in range(len(dep_adj_matrix[target_index])):
                if dep_adj_matrix[target_index][first_order_index] == 0:
                    continue
                for second_order_index in range(len(dep_adj_matrix[first_order_index])):
                    if dep_adj_matrix[first_order_index][second_order_index] == 0:
                        continue
                    if second_order_index == target_index:
                        continue
                    if new_dep_adj_matrix[target_index][second_order_index] == 1:
                        continue
                    new_dep_adj_matrix[target_index][second_order_index] = 1
                    new_dep_type_matrix[target_index][second_order_index] = dep_type_matrix[first_order_index][second_order_index]
        return new_dep_adj_matrix, new_dep_type_matrix

    def get_second_order(self, direct=False):
        dep_adj_matrix, dep_type_matrix = self.get_first_order(direct=direct)
        return self.get_next_order(dep_adj_matrix, dep_type_matrix)

    def get_third_order(self, direct=False):
        dep_adj_matrix, dep_type_matrix = self.get_second_order(direct=direct)
        return self.get_next_order(dep_adj_matrix, dep_type_matrix)

    def search_dep_path(self, start_idx, end_idx, adj_max, dep_path_arr):
        for next_id in range(len(adj_max[start_idx])):
            if next_id in dep_path_arr or adj_max[start_idx][next_id] in ["none"]:
                continue
            if next_id == end_idx:
                return 1, dep_path_arr + [next_id]
            stat, dep_arr = self.search_dep_path(next_id, end_idx, adj_max, dep_path_arr + [next_id])
            if stat == 1:
                return stat, dep_arr
        return 0, []

    def get_dep_path(self, start_index, end_index, direct=False):
        dep_adj_matrix, dep_type_matrix = self.get_first_order(direct=direct)
        _, dep_path = self.search_dep_path(start_index, end_index, dep_type_matrix, [start_index])
        return dep_path

class ABSADataset(Dataset):
    def __init__(self, datafile, tokenizer, opt, deptype2id=None, dep_order="first"):
        self.datafile = datafile
        self.depfile = "{}.dep".format(datafile)
        self.tokenizer = tokenizer
        self.opt = opt
        self.deptype2id = deptype2id
        self.dep_order = dep_order
        self.textdata = ABSADataset.load_datafile(self.datafile)
        self.depinfo = ABSADataset.load_depfile(self.depfile)
        self.polarity2id = self.get_polarity2id()
        self.feature = []
        for sentence,depinfo in zip(self.textdata, self.depinfo):
            self.feature.append(self.create_feature(sentence, depinfo))
        print(self.feature[:1])

    def __getitem__(self, index):
        return self.feature[index]

    def __len__(self):
        return len(self.feature)

    def ws(self, text):
        tokens = []
        valid_ids = []
        for i, word in enumerate(text):
            if len(text) <= 0:
                continue
            token = self.tokenizer.tokenizer.tokenize(word)
            tokens.extend(token)
            for m in range(len(token)):
                if m == 0:
                    valid_ids.append(1)
                else:
                    valid_ids.append(0)
        token_ids = self.tokenizer.tokenizer.convert_tokens_to_ids(tokens)
        return tokens, token_ids, valid_ids

    def create_feature(self, sentence, depinfo):
        text_left, text_right, aspect, polarity = sentence

        cls_id = self.tokenizer.tokenizer.vocab["[CLS]"]
        sep_id = self.tokenizer.tokenizer.vocab["[SEP]"]

        doc = text_left + " " + aspect + " " + text_right

        left_tokens, left_token_ids, left_valid_ids = self.ws(text_left.split(" "))
        right_tokens, right_token_ids, right_valid_ids = self.ws(text_right.split(" "))
        aspect_tokens, aspect_token_ids, aspect_valid_ids = self.ws(aspect.split(" "))
        tokens = left_tokens + aspect_tokens + right_tokens
        input_ids = [cls_id] + left_token_ids + aspect_token_ids + right_token_ids + [sep_id] + aspect_token_ids + [sep_id]
        valid_ids = [1] + left_valid_ids + aspect_valid_ids + right_valid_ids + [1] + aspect_valid_ids + [1]
        mem_valid_ids = [0] + [0] * len(left_tokens) + [1] * len(aspect_tokens) + [0] * len(right_tokens)
        segment_ids = [0] * (len(tokens) + 2) + [1] * (len(aspect_tokens)+1)

        dep_instance_parser = DepInstanceParser(basicDependencies=depinfo, tokens=[])
        if self.dep_order == "first":
            dep_adj_matrix, dep_type_matrix = dep_instance_parser.get_first_order()
        elif self.dep_order == "second":
            dep_adj_matrix, dep_type_matrix = dep_instance_parser.get_second_order()
        elif self.dep_order == "third":
            dep_adj_matrix, dep_type_matrix = dep_instance_parser.get_third_order()
        else:
            raise ValueError()

        token_head_list = []
        for input_id, valid_id in zip(input_ids, valid_ids):
            if input_id == cls_id:
                continue
            if input_id == sep_id:
                break
            if valid_id == 1:
                token_head_list.append(input_id)

        input_ids = self.tokenizer.id_to_sequence(input_ids)
        valid_ids = self.tokenizer.id_to_sequence(valid_ids)
        segment_ids = self.tokenizer.id_to_sequence(segment_ids)
        mem_valid_ids = self.tokenizer.id_to_sequence(mem_valid_ids)

        size = input_ids.shape[0]

        # final_dep_adj_matrix = [[0] * size for _ in range(self.tokenizer.max_seq_len)]
        # final_dep_value_matrix = [[0] * size for _ in range(self.tokenizer.max_seq_len)]
        final_dep_adj_matrix = [[0] * size for _ in range(size)]
        final_dep_value_matrix = [[0] * size for _ in range(size)]
        for i in range(len(token_head_list)):
            for j in range(len(dep_adj_matrix[i])):
                if j >= size:
                    break
                final_dep_adj_matrix[i+1][j] = dep_adj_matrix[i][j]
                final_dep_value_matrix[i+1][j] = self.deptype2id[dep_type_matrix[i][j]]

        return {
            "input_ids":torch.tensor(input_ids),
            "valid_ids":torch.tensor(valid_ids),
            "segment_ids":torch.tensor(segment_ids),
            "mem_valid_ids":torch.tensor(mem_valid_ids),
            "dep_adj_matrix":torch.tensor(final_dep_adj_matrix),
            "dep_value_matrix":torch.tensor(final_dep_value_matrix),
            "polarity": self.polarity2id[polarity],
            "raw_text": doc,
            "aspect": aspect
        }


    @staticmethod
    def load_depfile(filename):
        data = []
        with open(filename, 'r') as f:
            dep_info = []
            for line in f:
                line = line.strip()
                if len(line) > 0:
                    items = line.split("\t")
                    dep_info.append({
                        "governor": int(items[0]),
                        "dependent": int(items[1]),
                        "dep": items[2],
                    })
                else:
                    if len(dep_info) > 0:
                        data.append(dep_info)
                        dep_info = []
            if len(dep_info) > 0:
                data.append(dep_info)
                dep_info = []
        return data

    @staticmethod
    def load_datafile(filename):
        data = []
        with open(filename, 'r') as f:
            lines = f.readlines()
            for i in range(0, len(lines), 3):
                text_left, _, text_right = [s.lower().strip() for s in lines[i].partition("$T$")]
                aspect = lines[i + 1].lower().strip()
                text_right = text_right.replace("$T$", aspect)
                polarity = lines[i + 2].strip()
                data.append([text_left, text_right, aspect, polarity])

        return data

    @staticmethod
    def load_deptype_map(opt):
        deptype_set = set()
        for filename in [opt.train_file, opt.test_file, opt.val_file]:
            filename = "{}.dep".format(filename)
            if os.path.exists(filename) is False:
                continue
            data = ABSADataset.load_depfile(filename)
            for dep_info in data:
                for item in dep_info:
                    deptype_set.add(item['dep'])
        deptype_map = {"none": 0}
        for deptype in sorted(deptype_set, key=lambda x:x):
            deptype_map[deptype] = len(deptype_map)
        return deptype_map

    @staticmethod
    def get_polarity2id():
        polarity_label = ["-1","0","1"]
        return dict([(label, idx) for idx,label in enumerate(polarity_label)])


