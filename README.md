# PPGAN
Source code and datasets for paper: Selling Products by Machine: a User-Sensitive Adversarial Training method for Short Title Generation in Mobile E-Commerce.

## Dependencies
Python>=3.5
Tensorflow>=1.8.0
scikit-learn
numpy

## Datasets
There are five files in `data/` directory.
- click0.csv: user-clicked training set.
- click1.csv: user-unclicked training set.
- test.csv: testing set.
- vocab_gan.txt: Word list by word frequency statistics.

File format of "click0.csv", "click1.csv" and "test.csv":id, item_id, if_click, long_title, short_title, short_title_position_index, user_feature.

## Running
- train: run *train.py*
- test: run *tester.py*

## Citation
Paper：
Selling Products by Machine: a User-Sensitive Adversarial Training method for Short Title Generation in Mobile E-Commerce
