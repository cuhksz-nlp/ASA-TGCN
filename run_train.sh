#!/bin/bash


#LAPTOP
python asa_tgcn_main.py --do_train --train_file ./data/laptop/train.txt --test_file ./data/laptop/test.txt --bert_model ./bert-large-uncased/ --num_epoch 10 --model_name ASA_TGCN.LAPTOP.BERT.L --batch_size 32 --learning_rate 1e-5 --outdir ./

#REST14
python asa_tgcn_main.py --do_train --train_file ./data/rest14/train.txt --test_file ./data/rest14/test.txt --bert_model ./bert-large-uncased/ --num_epoch 10 --model_name ASA_TGCN.REST14.BERT.L --batch_size 32 --learning_rate 1e-5 --outdir ./

#REST15
python asa_tgcn_main.py --do_train --train_file ./data/rest15/train.txt --test_file ./data/rest15/test.txt --bert_model ./bert-large-uncased/ --num_epoch 10 --model_name ASA_TGCN.REST15.BERT.L --batch_size 32 --learning_rate 1e-5 --outdir ./

#REST16
python asa_tgcn_main.py --do_train --train_file ./data/rest16/train.txt --test_file ./data/rest16/test.txt --bert_model ./bert-large-uncased/ --num_epoch 10 --model_name ASA_TGCN.REST16.BERT.L --batch_size 32 --learning_rate 1e-5 --outdir ./

#TWITTER
python asa_tgcn_main.py --do_train --train_file ./data/twitter/train.txt --test_file ./data/twitter/test.txt --bert_model ./bert-large-uncased/ --num_epoch 10 --model_name ASA_TGCN.TWITTER.BERT.L --batch_size 32 --learning_rate 1e-5 --outdir ./

#MAMS
python asa_tgcn_main.py --do_train --train_file ./data/mams/train.txt --test_file ./data/mams/test.txt --bert_model ./bert-large-uncased/ --num_epoch 10 --model_name ASA_TGCN.MAMS.BERT.L --batch_size 32 --learning_rate 1e-5 --outdir ./
