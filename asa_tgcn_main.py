import logging
import argparse
import math
import os
import sys
from time import strftime, localtime
import random
import numpy as np
import subprocess

from pytorch_transformers import BertModel, BertConfig
from data_utils import Tokenizer4Bert, ABSADataset
from asa_tgcn_model import AsaTgcn

from sklearn import metrics
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, random_split


CONFIG_NAME = 'config.json'
WEIGHTS_NAME = 'pytorch_model.bin'

logger = logging.getLogger()
logger.setLevel(logging.INFO)
logger.addHandler(logging.StreamHandler(sys.stdout))


class Instructor:
    def __init__(self, opt):
        self.opt = opt
        logger.info(opt)
        deptype2id = ABSADataset.load_deptype_map(opt)
        polarity2id = ABSADataset.get_polarity2id()
        logger.info(deptype2id)
        logger.info(polarity2id)
        self.deptype2id = deptype2id
        self.polarity2id = polarity2id

        self.vocab_path = os.path.join(opt.bert_model, 'vocab.txt')
        self.tokenizer = Tokenizer4Bert(opt.max_seq_len, opt.bert_model)
        config = BertConfig.from_json_file(os.path.join(opt.bert_model, CONFIG_NAME))
        config.num_labels=opt.polarities_dim
        config.num_types=len(self.deptype2id)
        logger.info(config)
        self.model = AsaTgcn.from_pretrained(opt.bert_model, config=config)
        self.model.to(opt.device)

        self.trainset = ABSADataset(opt.train_file, self.tokenizer, self.opt, deptype2id=deptype2id)
        self.testset = ABSADataset(opt.test_file, self.tokenizer, self.opt, deptype2id=deptype2id)
        if os.path.exists(opt.val_file):
            self.valset = ABSADataset(opt.val_file, self.tokenizer, self.opt, deptype2id=deptype2id)
        elif opt.valset_ratio > 0:
            valset_len = int(len(self.trainset) * opt.valset_ratio)
            self.trainset, self.valset = random_split(self.trainset, (len(self.trainset)-valset_len, valset_len))
        else:
            self.valset = self.testset

        if opt.device.type == 'cuda':
            logger.info('cuda memory allocated: {}'.format(torch.cuda.memory_allocated(device=opt.device.index)))

    def _print_args(self):
        n_trainable_params, n_nontrainable_params = 0, 0
        for p in self.model.parameters():
            n_params = torch.prod(torch.tensor(p.shape))
            if p.requires_grad:
                n_trainable_params += n_params
            else:
                n_nontrainable_params += n_params
        logger.info('n_trainable_params: {0}, n_nontrainable_params: {1}'.format(n_trainable_params, n_nontrainable_params))
        logger.info('> training arguments:')
        for arg in vars(self.opt):
            logger.info('>>> {0}: {1}'.format(arg, getattr(self.opt, arg)))

    def _reset_params(self):
        for child in self.model.children():
            if type(child) != BertModel:  # skip bert params
                for p in child.parameters():
                    if p.requires_grad:
                        if len(p.shape) > 1:
                            torch.nn.init.xavier_uniform_(p)
                        else:
                            stdv = 1. / math.sqrt(p.shape[0])
                            torch.nn.init.uniform_(p, a=-stdv, b=stdv)

    def save_model(self, save_path, model, args):
        # Save a trained model, configuration and tokenizer
        model_to_save = model.module if hasattr(model, 'module') else model  # Only save the model it-self
        # If we save using the predefined names, we can load using `from_pretrained`
        output_model_file = os.path.join(save_path, WEIGHTS_NAME)
        output_config_file = os.path.join(save_path, CONFIG_NAME)
        torch.save(model_to_save.state_dict(), output_model_file)

        config = model_to_save.config
        config.__dict__["deptype2id"] = self.deptype2id
        config.__dict__["polarity2id"] = self.polarity2id
        with open(output_config_file, "w", encoding='utf-8') as writer:
            writer.write(config.to_json_string())
        output_args_file = os.path.join(save_path, 'training_args.bin')
        torch.save(args, output_args_file)
        subprocess.run(['cp', self.vocab_path, os.path.join(save_path, 'vocab.txt')])

    def _train(self, criterion, optimizer, train_data_loader, val_data_loader, test_data_loader):
        max_val_acc = -1
        global_step = 0
        path = None

        model_home = self.opt.model_path + '-' + strftime("%y%m%d-%H%M", localtime())

        results = {"bert_model": self.opt.bert_model, "batch_size": self.opt.batch_size,
                   "learning_rate": self.opt.learning_rate, "seed": self.opt.seed}
        for epoch in range(self.opt.num_epoch):
            logger.info('>' * 100)
            logger.info('epoch: {}'.format(epoch))
            n_correct, n_total, loss_total = 0, 0, 0
            self.model.train()
            for i_batch, t_sample_batched in enumerate(train_data_loader):
                global_step += 1
                optimizer.zero_grad()
                outputs = self.model(t_sample_batched["input_ids"].to(self.opt.device),
                                     t_sample_batched["segment_ids"].to(self.opt.device),
                                     t_sample_batched["valid_ids"].to(self.opt.device),
                                     t_sample_batched["mem_valid_ids"].to(self.opt.device),
                                     t_sample_batched["dep_adj_matrix"].to(self.opt.device),
                                     t_sample_batched["dep_value_matrix"].to(self.opt.device))

                targets = t_sample_batched['polarity'].to(self.opt.device)

                loss = criterion(outputs, targets)
                loss.backward()

                optimizer.step()

                n_correct += (torch.argmax(outputs, -1) == targets).sum().item()
                n_total += len(outputs)
                loss_total += loss.item() * len(outputs)
                if global_step % self.opt.log_step == 0:
                    train_acc = n_correct / n_total
                    train_loss = loss_total / n_total
                    logger.info('epoch: {}, loss: {:.4f}, train acc: {:.4f}'.format(epoch, train_loss, train_acc))
            val_acc, val_f1 = Instructor._evaluate_acc_f1(self.model, val_data_loader, device=self.opt.device)
            logger.info('>epoch: {}, val_acc: {:.4f}, val_f1: {:.4f}'.format(epoch, val_acc, val_f1))
            results["{}_val_acc".format(epoch)] = val_acc
            results["{}_val_f1".format(epoch)] = val_f1

            if val_acc > max_val_acc:
                max_val_acc = val_acc
                saving_path = os.path.join(model_home, "epoch_{}".format(epoch))
                if not os.path.exists(saving_path):
                    os.makedirs(saving_path)
                self.save_model(saving_path, self.model, self.opt)

                self.model.eval()
                saving_path = os.path.join(model_home, "epoch_{}_eval.txt".format(epoch))
                test_acc, test_f1 = self._evaluate_acc_f1(self.model, test_data_loader, device=self.opt.device,
                                                          saving_path=saving_path)
                logger.info('>> epoch: {}, test_acc: {:.4f}, test_f1: {:.4f}'.format(epoch, test_acc, test_f1))

                results["max_val_acc"] = max_val_acc
                results["test_acc"] = test_acc
                results["test_f1"] = test_f1

            output_eval_file = os.path.join(model_home, "eval_results.txt")
            with open(output_eval_file, "w") as writer:
                for k,v in results.items():
                    writer.write("{}={}\n".format(k,v))
        return path

    @staticmethod
    def _evaluate_acc_f1(model, data_loader, device, saving_path=None):
        n_correct, n_total = 0, 0
        t_targets_all, t_outputs_all = None, None
        model.eval()

        saving_path_f = open(saving_path, 'w') if saving_path is not None else None

        with torch.no_grad():
            for t_batch, t_sample_batched in enumerate(data_loader):
                t_targets = t_sample_batched['polarity'].to(device)
                t_raw_texts = t_sample_batched['raw_text']
                t_aspects = t_sample_batched['aspect']

                t_outputs = model(t_sample_batched["input_ids"].to(device),
                                  t_sample_batched["segment_ids"].to(device),
                                  t_sample_batched["valid_ids"].to(device),
                                  t_sample_batched["mem_valid_ids"].to(device),
                                  t_sample_batched["dep_adj_matrix"].to(device),
                                  t_sample_batched["dep_value_matrix"].to(device))

                n_correct += (torch.argmax(t_outputs, -1) == t_targets).sum().item()
                n_total += len(t_outputs)

                if t_targets_all is None:
                    t_targets_all = t_targets
                    t_outputs_all = t_outputs
                else:
                    t_targets_all = torch.cat((t_targets_all, t_targets), dim=0)
                    t_outputs_all = torch.cat((t_outputs_all, t_outputs), dim=0)

                if saving_path_f is not None:
                    for t_target, t_output, t_raw_text, t_aspect in zip(t_targets.detach().cpu().numpy(),
                                                                        torch.argmax(t_outputs, -1).detach().cpu().numpy(),
                                                                        t_raw_texts, t_aspects):
                        saving_path_f.write("{}\t{}\t{}\t{}\n".format(t_target, t_output, t_raw_text, t_aspect))

        acc = n_correct / n_total
        f1 = metrics.f1_score(t_targets_all.cpu(), torch.argmax(t_outputs_all, -1).cpu(), labels=[0, 1, 2], average='macro')
        return acc, f1

    def train(self):
        # Loss and Optimizer
        criterion = nn.CrossEntropyLoss()
        _params = filter(lambda p: p.requires_grad, self.model.parameters())
        optimizer = torch.optim.Adam(_params, lr=self.opt.learning_rate, weight_decay=self.opt.l2reg)

        train_data_loader = DataLoader(dataset=self.trainset, batch_size=self.opt.batch_size, shuffle=True)
        test_data_loader = DataLoader(dataset=self.testset, batch_size=self.opt.batch_size, shuffle=False)
        val_data_loader = DataLoader(dataset=self.valset, batch_size=self.opt.batch_size, shuffle=False)

        self._reset_params()
        self._train(criterion, optimizer, train_data_loader, val_data_loader, test_data_loader)


def test(opt):
    logger.info(opt)
    config = BertConfig.from_json_file(os.path.join(opt.model_path, CONFIG_NAME))
    logger.info(config)

    tokenizer = Tokenizer4Bert(opt.max_seq_len, opt.model_path)
    model = AsaTgcn.from_pretrained(opt.model_path)
    model.to(opt.device)

    deptype2id = config.deptype2id
    logger.info(deptype2id)
    testset = ABSADataset(opt.test_file, tokenizer, opt, deptype2id=deptype2id)
    test_data_loader = DataLoader(dataset=testset, batch_size=opt.batch_size, shuffle=False)
    test_acc, test_f1 = Instructor._evaluate_acc_f1(model, test_data_loader, device=opt.device)
    logger.info('>> test_acc: {:.4f}, test_f1: {:.4f}'.format(test_acc, test_f1))


def get_args():
    # Hyper Parameters
    parser = argparse.ArgumentParser()
    parser.add_argument('--train_file', default='sample_data/train.txt', type=str)
    parser.add_argument('--test_file', default='sample_data/test.txt', type=str)
    parser.add_argument('--val_file', default='sample_data/val.txt', type=str)
    parser.add_argument('--log', default='log', type=str)
    parser.add_argument('--bert_model', default='./bert-large-uncased', type=str)
    parser.add_argument('--model_path', default='./models/tmp_model', type=str)
    parser.add_argument('--learning_rate', default='2e-5', type=float)
    parser.add_argument('--dropout', default=0, type=float)
    parser.add_argument('--bert_dropout', default=0.2, type=float)
    parser.add_argument('--l2reg', default=0.01, type=float)
    parser.add_argument('--num_epoch', default=30, type=int)
    parser.add_argument('--batch_size', default=16, type=int)
    parser.add_argument('--log_step', default=5, type=int)
    parser.add_argument('--max_seq_len', default=100, type=int)
    parser.add_argument('--polarities_dim', default=3, type=int)
    parser.add_argument('--device', default=None, type=str)
    parser.add_argument('--seed', default=50, type=int)
    parser.add_argument('--valset_ratio', default=0, type=float)
    parser.add_argument("--do_train", action='store_true', help="Whether to run training.")
    parser.add_argument("--do_eval", action='store_true', help="Whether to run eval on the dev set.")
    opt = parser.parse_args()

    return opt


def set_seed(opt):
    if opt.seed is not None:
        random.seed(opt.seed)
        np.random.seed(opt.seed)
        torch.manual_seed(opt.seed)
        torch.cuda.manual_seed(opt.seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


def main():
    opt = get_args()
    set_seed(opt)

    opt.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu') \
        if opt.device is None else torch.device(opt.device)
    opt.n_gpu = torch.cuda.device_count()

    if not os.path.exists(opt.log):
        os.makedirs(opt.log)

    log_file = '{}/log-{}.log'.format(opt.log, strftime("%y%m%d-%H%M", localtime()))
    logger.addHandler(logging.FileHandler(log_file))

    if opt.do_train:
        ins = Instructor(opt)
        ins.train()
    elif opt.do_eval:
        test(opt)


if __name__ == '__main__':
    main()
