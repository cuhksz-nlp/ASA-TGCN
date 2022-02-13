DATA = $1

for VARIABLE in 1 2 3 4 .. 20
do
    python asa_tgcn_main.py --do_train --do_eval --train_file ./data/data/$DATA/train.txt --test_file ./data/data/$DATA/test.txt --val_file ./data/data/$DATA/test.txt --bert_model ./bert_large_uncased --do_lower_case --num_epoch 50 --batch_size 4 --learning_rate 1e-5 --model_path ./models/$DATA/ASA_TGCN.BERT.L.$VARIABLE --seed=$VARIABLE
done
