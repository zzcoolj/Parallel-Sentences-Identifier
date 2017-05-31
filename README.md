# zNLP : Identifying parallel sentences in Chinese-English comparable corpora for the [BUCC 2017 Shared Task](https://comparable.limsi.fr/bucc2017/bucc2017-task.html)

## Directory layout
```
.
├── code
└── data
    ├── bucc2017
    │   ├── test_data
    │   │   ├── zh-en.test.en
    │   │   └── zh-en.test.zh
    │   └── training_data
    │       ├── zh-en.training.en
    │       ├── zh-en.training.gold
    │       └── zh-en.training.zh
    ├── dictionaries
    ├── stopwords
    └── temp_data
        └── classifier
            ├── test
            └── training
```
test_data and training_data could be downloaded from [**BUCC 2017 Shared Task web page**](https://comparable.limsi.fr/bucc2017/bucc2017-task.html). As for the dictionaries, I use [**CC-CEDICT**](https://www.mdbg.net/chinese/dictionary?page=cedict) and the restricted data `Chinese-English Translation Lexicon Version 3.0 [LDC2002L27]` (Huang et al., 2002) to generate Chinese-English dictionaries (generating functions are provided in `ChineseEnglishDictionary` class in `chinese_corpus_translator.py`). You could generate your own dictionaries by using other resources and then configure the dictionaries path in `config.ini`.


## Benchmark
Precision | Recall | F1-score | Remark
--------- | ------ | -------- | ------
0.4247 | 0.4815 | 0.4513 | From functional programming to OOP; Debugs
0.4242 | 0.4441 | 0.4339 | Baseline (First public version for paper review)
