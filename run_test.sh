#!/bin/bash

#LAPTOP
srun python asa_tgcn_main.py --do_eval --model_path ./release/ASA_TGCN.LAPTOP.BERT.L/ --test_file ./data/laptop/test.txt

#REST14
srun python asa_tgcn_main.py --do_eval --model_path ./release/ASA_TGCN.REST14.BERT.L/ --test_file ./data/rest14/test.txt

#REST15
srun python asa_tgcn_main.py --do_eval --model_path ./release/ASA_TGCN.REST15.BERT.L/ --test_file ./data/rest15/test.txt

#REST16
srun python asa_tgcn_main.py --do_eval --model_path ./release/ASA_TGCN.REST16.BERT.L/ --test_file ./data/rest16/test.txt

#TWITTER
srun python asa_tgcn_main.py --do_eval --model_path ./release/ASA_TGCN.TWITTER.BERT.L/ --test_file ./data/twitter/test.txt

#MAMS
srun python asa_tgcn_main.py --do_eval --model_path ./release/ASA_TGCN.MAMS.BERT.L/ --test_file ./data/mams/test.txt
