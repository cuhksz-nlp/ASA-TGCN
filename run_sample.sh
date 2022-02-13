#train
python asa_tgcn_main.py --do_train --do_eval --train_file ./data/sample_data/train.txt --test_file ./data/sample_data/test.txt --val_file ./data/sample_data/val.txt --bert_model ./bert_base_uncased --num_epoch 2 --batch_size 1 --learning_rate 1e-5 --model_path ./models/ASA_TGCN.SAMPLE.BERT.L
#test
python asa_tgcn_main.py --do_eval --model_path ./models/ASA_TGCN.SAMPLE.BERT.L/ --test_file ./data/sample_data/test.txt
