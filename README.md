# ASA-TGCN

This is the implementation of [Aspect-based Sentiment Analysis withType-aware Graph Convolutional Networks and Layer Ensemble](https://www.aclweb.org/anthology/2021.naacl-main.231/) at NAACL 2021.

You can e-mail Yuanhe Tian at `yhtian@uw.edu`, if you have any questions.

## Citation

If you use or extend our work, please cite our paper at NAACL 2021.

```
@inproceedings{tian-etal-2021-aspect,
    title = "Aspect-based Sentiment Analysis with Type-aware Graph Convolutional Networks and Layer Ensemble",
    author = "Tian, Yuanhe and Chen, Guimin and Song, Yan",
    booktitle = "Proceedings of the 2021 Conference of the North American Chapter of the Association for Computational Linguistics: Human Language Technologies",
    month = jun,
    year = "2021",
    address = "Online",
    publisher = "Association for Computational Linguistics",
    pages = "2910--2922"
}
```

## Requirements

Our code works with the following environment.
* `python=3.7`
* `pytorch=1.3`

## Dataset

To obtain the data, you can go to [`data`](./data) directory for details.

## Downloading BERT and ASA-TGCN

In our paper, we use BERT ([paper](https://www.aclweb.org/anthology/N19-1423/)) as the encoder.

For BERT, please download pre-trained BERT-Base and BERT-Large English from [Google](https://github.com/google-research/bert) or from [HuggingFace](https://s3.amazonaws.com/models.huggingface.co/bert/bert-base-chinese.tar.gz). If you download it from Google, you need to convert the model from TensorFlow version to PyTorch version.

[comment]: <> (For ASA-TGCN, you can download the models we trained in our experiments from [Google Drive] or [Baidu Net Disk].)

## Training and Testing on Sample Data

Run `run_sample.sh` to train a model on the small sample data under the `sample_data` directory.

Here are some important parameters:

* `--do_train`: train the model.
* `--do_eval`: test the model.

## To-do List

* Release the models.
* Regular maintenance.

You can leave comments in the `Issues` section, if you want us to implement any functions.

