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
test_data and training_data could be downloaded from [**BUCC 2017 Shared Task web page**](https://comparable.limsi.fr/bucc2017/bucc2017-task.html). As for the dictionaries, I use [**CC-CEDICT**](https://www.mdbg.net/chinese/dictionary?page=cedict) and the restricted data `Chinese-English Translation Lexicon Version 3.0 [LDC2002L27]` (Huang et al., 2002) to generate Chinese-English dictionaries (generating functions are provided in `ChineseEnglishDictionary` class of `chinese_corpus_translator.py`). You could also generate your own dictionaries by using other resources and then don't forget to configure the dictionaries path in `config.ini`.


## Benchmark
Precision | Recall | F1-score | Remark
--------- | ------ | -------- | ------
0.4242 | 0.4441 | 0.4339 | Baseline (First public version for paper review)
0.4247 | 0.4815 | 0.4513 | From functional programming to OOP; Debugs
0.4293 | 0.5348 | 0.4763 | `solr_topN` changed from 3 to 1; Remove `Solr_index` feature (as it is always 1); New overlap function
0.4370 | 0.5506 | 0.4873 | Independent corpus for overlap calculation: search engine Chinese tokenizer mode (full mode for the Solr searching corpus), remove English stop words and do English stemming.
