import nltk
import re
import json
import string
import jieba
import configparser
import common
from sklearn.externals import joblib

config = configparser.ConfigParser()
config.read('config.ini')


class ChineseCorpusTranslator(object):
    def __init__(self, zh_en_dict_path,
                 remove_chinese_stopwords, english_remove_stopwords,
                 english_stem, english_stem_for_dict,
                 tokenizer_mode='full_mode',):
        self.chinese_sentence_translator = ChineseSentenceTranslator(
            zh_en_dict_path=zh_en_dict_path,
            tokenizer_mode=tokenizer_mode,
            remove_chinese_stopwords=remove_chinese_stopwords,
            english_remove_stopwords=english_remove_stopwords,
            english_stem=english_stem,
            english_stem_for_dict=english_stem_for_dict)

    def translate(self, corpus_file_path, unknown_words_path,
                  translated_corpus_path, translated_corpus_for_selecter_path, translated_corpus_list_of_list_path):
        translated_target_file_dict = {}
        target_file_word_count_dict = {}
        tokens_translations_dict = {}
        all_unknown_words = set()
        corpus = common.read_two_columns_file_to_build_dictionary_type_specified(corpus_file_path, str, str)

        for identifier, sentence in corpus.items():
            sentence_translation, unknown_words, tokens_translations = self.chinese_sentence_translator.translate(sentence)
            all_unknown_words |= unknown_words
            translated_target_file_dict[identifier] = sentence_translation
            target_file_word_count_dict[identifier] = ChineseSentenceTranslator.get_chinese_sentence_length(sentence)
            tokens_translations_dict[identifier] = tokens_translations

        # Write unknown words
        unknown_words_file = open(unknown_words_path, 'w')
        for item in list(all_unknown_words):
            unknown_words_file.write("%s\n" % item)
        # Write translated target file to a file for Solr indexing and a file for selecter(SVM)
        f = open(translated_corpus_path, 'w')
        f2 = open(translated_corpus_for_selecter_path, 'w')
        for identifier, sentence_translation in translated_target_file_dict.items():
            f.write('%s\t%s\n' % (identifier, sentence_translation))
            # i.e. id,english_translation,original(Chinese)_sentence_length(count by words)
            f2.write('%s,%s,%d\n' % (identifier, sentence_translation, target_file_word_count_dict[identifier]))
        # Write tokens_translations_dict
        joblib.dump(tokens_translations_dict, translated_corpus_list_of_list_path)


class ChineseSentenceTranslator(object):
    def __init__(self, zh_en_dict_path, tokenizer_mode,
                 remove_chinese_stopwords, english_remove_stopwords,
                 english_stem, english_stem_for_dict):
        self.zh_en_dict = common.read_n_columns_file_to_build_dict(zh_en_dict_path)
        self.remove_chinese_stopwords = remove_chinese_stopwords
        self.english_remove_stopwords = english_remove_stopwords
        self.english_stem = english_stem
        self.english_stem_for_dict = english_stem_for_dict
        self.chinese_tokenizer = ChineseSentenceTranslator.get_chinese_tokenizer(tokenizer_mode)

        print(self.remove_chinese_stopwords)
        exit()

        # Prepare resources for removing stop words and stem.
        if english_remove_stopwords:
            self.stpwds = set(nltk.corpus.stopwords.words("english"))
        if english_stem or english_stem_for_dict:
            self.stemmer = nltk.stem.PorterStemmer()
        if remove_chinese_stopwords:
            # read chinese stopwords
            chinese_stpwds = set()
            with open(config['chinese_stopwords_resources']['github_6_stopwords_json']) as json_data:
                d = json.load(json_data)
            chinese_stpwds |= set(d)
            with open(config['chinese_stopwords_resources']['github_stopwords_iso']) as stopwords_file1:
                content = stopwords_file1.readlines()
            content = [x.strip() for x in content]
            chinese_stpwds |= set(content)
            with open(config['chinese_stopwords_resources']['githubgist_dreampuf']) as stopwords_file2:
                content2 = stopwords_file2.readlines()
            content2 = [x.strip() for x in content2]
            chinese_stpwds |= set(content2)
            self.chinese_stopwords = chinese_stpwds

    def translate(self, sentence):
        # Get sentence segmentation
        chinese_tokens_list = list(self.chinese_tokenizer(sentence))
        chinese_tokens_list = self.__preprocess_chinese_sentence(chinese_tokens_list)
        # list of string, cause each word may have several translations
        tokens_translations = []
        unknown_words = set()

        for chinese_token in chinese_tokens_list:
            # Avoid empty token (In full mode, token could be empty.)
            if chinese_token:
                # Check whether the target word in dictionary
                if chinese_token in self.zh_en_dict:
                    token_translations = self.zh_en_dict[chinese_token]
                    # TODO LATER interesting point & compare results
                    if self.english_stem_for_dict:
                        token_translations = list(set([self.stemmer.stem(e) for e in token_translations]))

                    tokens_translations.append(self.__process_token_translations(token_translations))
                else:
                    try:
                        # check whether the string can be encoded only with ASCII characters
                        # (which are Latin alphabet + some other characters)
                        chinese_token.encode('ascii')
                    except UnicodeEncodeError:
                        unknown_words.add(chinese_token)
                    else:
                        tokens_translations.append(self.__process_token_translations([chinese_token]))

        sentence_translation = ' '.join(tokens_translations)

        # TODO Useless, after test OK => discard
        # # Before loading csv file into Solr, remove ',' and '"'
        # # Attention : same code in __init__ of CandidateParallelSentencePairsFinder
        # sentence_translation = sentence_translation.replace(',', ' ')
        # sentence_translation = sentence_translation.replace('"', ' ')

        return sentence_translation, unknown_words, tokens_translations

    def __preprocess_chinese_sentence(self, chinese_tokens_list):
        if self.remove_chinese_stopwords:
            result = [chinese_token for chinese_token in chinese_tokens_list if
                      chinese_token not in self.chinese_stopwords]
            return result
        return chinese_tokens_list

    def __process_token_translations(self, token_translations):
        """
        :param token_translations: list of translations of a Chinese token
        :return: string (concatenation of all the translations)
        """
        # an element of token_translations e.g. (conjunction used to express contrast with a previous clause) but
        # Note that one Chinese word's translation may contain more than one word & may contain punctuation marks.
        # => Concatenate all translations into one string & remove punctuation marks
        units = ' '.join(token_translations).lower().translate(str.maketrans('', '', string.punctuation)).split(' ')
        processed_units = []
        for unit in units:
            # replace consecutive non-ASCII characters with a space
            unit = re.sub(r'[^\x00-\x7F]+', ' ', unit)
            if self.english_remove_stopwords and (unit in self.stpwds):
                continue
            if self.english_stem:
                unit = self.stemmer.stem(unit)
            processed_units.append(unit)
            # TODO LATER Use all potential translations so far, maybe only use the first one?
        return ' '.join(processed_units)

    @staticmethod
    def get_chinese_tokenizer(mode):
        def tokenizer_full_mode(sentence):
            result = jieba.cut(sentence, cut_all=True)
            return result

        def tokenizer_accurate_mode(sentence):
            return jieba.cut(sentence, cut_all=False)

        def tokenizer_search_engine_mode(sentence):
            return jieba.cut_for_search(sentence)

        if mode is 'full_mode':
            return tokenizer_full_mode
        elif mode is 'accurate_mode':
            return tokenizer_accurate_mode
        elif mode is 'search_engine_mode':
            return tokenizer_search_engine_mode
        else:
            raise ValueError("Don't have this kind of tokenizer mode.")

    @staticmethod
    def get_chinese_sentence_length(sent):
        # Sentence length should be always been calculated in accurate mode
        words = jieba.cut(sent, cut_all=False)
        return len(list(words))

    def get_average_translations_num_per_word(self):
        translation_num = []
        for value in self.zh_en_dict.values():
            translation_num.append(len(value))
        return sum(translation_num) / len(translation_num)


class Dictionary(object):
    @staticmethod
    def merge_two_dictionaries(d1, d2):
        for key, value in d2.items():
            if key in d1:
                d1[key] = list(set(d1[key] + value))
            else:
                d1[key] = value
        return d1


class ChineseEnglishDictionary(Dictionary):
    def __init__(self):
        pass

    def update_dictionary(self, old_dictionary_path, output_path):
        result = common.read_n_columns_file_to_build_dict(old_dictionary_path)
        unknown_words_dict = self.load_unknown_words_dict(
            zh_path='/Users/zzcoolj/Code/bucc2017/data/dictionaries/unknown_words.txt',
            en_path='/Users/zzcoolj/Code/bucc2017/data/dictionaries/unknown_words.en.txt'
        )
        result = self.merge_two_dictionaries(result, unknown_words_dict)
        common.write_dict_to_file_value_type_specified(output_path, result, list)

    @staticmethod
    def load_cc_cedict_data(file_path):
        # 中國證監會 中国证监会 [Zhong1 guo2 Zheng4 jian4 hui4] /China Securities Regulatory Commission (CSRC)/abbr. for 中國證券監督管理委員會|中国证券监督管理委员会/
        d = {}
        with open(file_path) as f:
            for line in f:
                # Skip comment line
                if line[0] in ['#', '%']:
                    continue
                line = line.rstrip('\n')
                chinese_simplified = line.split(" ")[1]
                chinese_traditional = line.split(" ")[0]
                # One chinese word may have more than one english translations
                english = line[line.find('/') + 1:-1]
                english_translations = english.split('/')
                # 中文简繁体都有
                d[chinese_simplified] = english_translations
                d[chinese_traditional] = english_translations
        return d

    @staticmethod
    def load_ldc_cedict_gb_v3(file_path):
        # sample (encoded with GB2312):   阿波罗	/Apollo/
        d = {}
        with open(file_path, "rb") as f:
            for line in f:
                # gb2312 encoding transfer to Unicode
                line_elements = line.decode('gb2312').rstrip('\n').split('\t')
                chinese_simplified = line_elements[0]
                english = line_elements[1][1:-1]
                # One chinese word may have more than one english translations
                english_translations = english.split('/')
                d[chinese_simplified] = english_translations
        return d

    @staticmethod
    def load_unknown_words_dict(zh_path, en_path):
        with open(en_path) as f:
            unknown_words_merged_accurate_mode_en = list(f.read().splitlines())
            unknown_words_merged_accurate_mode_en[0] = unknown_words_merged_accurate_mode_en[0].replace('\ufeff',
                                                                                                        '')
            print(len(unknown_words_merged_accurate_mode_en))

        with open(zh_path) as f:
            unknown_words_merged_accurate_mode_zh = list(f.read().splitlines())
            print(len(unknown_words_merged_accurate_mode_zh))

        merged_unknown_words_dict = dict(
            zip(unknown_words_merged_accurate_mode_zh, unknown_words_merged_accurate_mode_en))
        return merged_unknown_words_dict


class EnglishChineseDictionary(Dictionary):
    """
    Unlike in ChineseEnglishDictionary, Chinese word could be both simplified or traditional.
    In EnglishChineseDictionary, Chinese is always in simplified format.
    """

    @staticmethod
    def load_ldc_cedict_gb_v3(file_path):
        # sample (encoded with GB2312):   阿波罗	/Apollo/
        d = {}
        with open(file_path, "rb") as f:
            for line in f:
                # gb2312 encoding transfer to Unicode
                line_elements = line.decode('gb2312').rstrip('\n').split('\t')
                chinese_simplified = line_elements[0]
                english = line_elements[1][1:-1]
                # One chinese word may have more than one english translations
                english_translations = english.split('/')
                for english_translation in english_translations:
                    if english_translation in d:
                        d[english_translation].append(chinese_simplified)
                    else:
                        d[english_translation] = [chinese_simplified]
        return d

    @staticmethod
    def load_cc_cedict_data(file_path):
        # 中國證監會 中国证监会 [Zhong1 guo2 Zheng4 jian4 hui4] /China Securities Regulatory Commission (CSRC)/abbr. for 中國證券監督管理委員會|中国证券监督管理委员会/
        d = {}
        with open(file_path) as f:
            for line in f:
                # Skip comment line
                if line[0] in ['#', '%']:
                    continue
                line = line.rstrip('\n')
                chinese_simplified = line.split(" ")[1]
                # One chinese word may have more than one english translations
                english = line[line.find('/') + 1:-1]
                english_translations = english.split('/')
                for english_translation in english_translations:
                    if english_translation in d:
                        d[english_translation].append(chinese_simplified)
                    else:
                        d[english_translation] = [chinese_simplified]
        return d


class Corpus(object):
    def __init__(self):
        pass

    @staticmethod
    def set_bucc_sample_new_id():
        # zh-en.training.zh last id: zh-000094637
        sample_target_new_id_gap = 94637
        new_sample_target_file = open('/Users/zzcoolj/Code/bucc2017/data/bucc2017/sample_data/zh-en.sample.zh_new', 'w')
        target_id_dict = dict()
        with open('/Users/zzcoolj/Code/bucc2017/data/bucc2017/sample_data/zh-en.sample.zh') as f:
            for line in f:
                (old_id, sentence) = line.rstrip('\n').split("\t")
                new_id_value = str(int(old_id.split('-')[1]) + sample_target_new_id_gap)
                new_id = 'zh-' + '0' * (9 - len(new_id_value)) + new_id_value
                print(old_id, '=>', new_id)
                target_id_dict[old_id] = new_id
                new_sample_target_file.write('%s\t%s\n' % (new_id, sentence))

        # zh-en.sample.en last id: en-000088860
        sample_source_new_id_gap = 88860
        new_sample_source_file = open('/Users/zzcoolj/Code/bucc2017/data/bucc2017/sample_data/zh-en.sample.en_new', 'w')
        source_id_dict = dict()
        with open('/Users/zzcoolj/Code/bucc2017/data/bucc2017/sample_data/zh-en.sample.en') as f:
            for line in f:
                (old_id, sentence) = line.rstrip('\n').split("\t")
                new_id_value = str(int(old_id.split('-')[1]) + sample_source_new_id_gap)
                new_id = 'en-' + '0' * (9 - len(new_id_value)) + new_id_value
                print(old_id, '=>', new_id)
                source_id_dict[old_id] = new_id
                new_sample_source_file.write('%s\t%s\n' % (new_id, sentence))

        new_gold_file = open('/Users/zzcoolj/Code/bucc2017/data/bucc2017/sample_data/zh-en.sample.gold_new', 'w')
        with open('/Users/zzcoolj/Code/bucc2017/data/bucc2017/sample_data/zh-en.sample.gold') as f:
            for line in f:
                (target_id, source_id) = line.rstrip('\n').split("\t")
                new_gold_file.write(target_id_dict[target_id] + '\t' + source_id_dict[source_id] + '\n')

    @staticmethod
    def calculate_target_source_parallel_sentences_length_rate(source_file_path, target_file_path,
                                                               target_source_dict_path):
        lengths = []
        dict_source = common.read_two_columns_file_to_build_dictionary(source_file_path)
        dict_target = common.read_two_columns_file_to_build_dictionary(target_file_path)
        with open(target_source_dict_path) as f:
            for target_source in f:
                target_id, source_id = target_source.rstrip('\n').split("\t")
                len_source = len(dict_source[source_id].strip().split(' '))
                # accurate mode
                len_target = sum(1 for x in jieba.cut(dict_target[target_id].strip(), cut_all=False))
                lengths.append([len_target, len_source])
                # According to the statistics of training data length rate, it is rare that rate is bigger than 2.
                if len_target / len_source >= 2:
                    print(dict_source[source_id])
                    print(dict_target[target_id])
                    print(' '.join(jieba.cut(dict_target[target_id].strip(), cut_all=False)))
                    print('*******************')

        len_rate = [len_target / len_source for len_target, len_source in lengths]
        print(max(len_rate))
        print(min(len_rate))
        return lengths, len_rate
    @staticmethod
    def show_parallel_sentences_length_rate(len_rate):
        import numpy as np
        import scipy.stats as stats
        import pylab as pl

        h = sorted(len_rate)
        # probability density function
        # numpy.std: Compute the standard deviation along the specified axis.
        fit = stats.norm.pdf(h, np.mean(h), np.std(h))
        pl.plot(h, fit, '-o')
        pl.hist(h, normed=True)
        pl.title('Parallel sentences (Chinese - English) length rate')
        pl.show()


# # Sentence translation test
# cct = ChineseCorpusTranslator()
# sentence_translation, unknown_words = cct.chinese_sentence_translator.translate('輪狀病毒則通常是通過與被感染的兒童的直接接觸傳播。')
# print(sentence_translation)

# # Build English-Chinese dictionary
# d = EnglishChineseDictionary.merge_two_dictionaries(
#     EnglishChineseDictionary.load_ldc_cedict_gb_v3('/Users/zzcoolj/Code/bucc2017/data/dictionaries/ldc2002l27/data/ldc_cedict.gb.v3'),
#     EnglishChineseDictionary.load_cc_cedict_data('/Users/zzcoolj/Code/bucc2017/data/dictionaries/cedict_ts.u8'))
# common.write_dict_to_file_value_type_specified('/Users/zzcoolj/Code/bucc2017/data/dictionaries/en_ch_dict_cc_ldc', d, list)
