import chinese_corpus_translator
import candidate_parallel_sentence_pairs_searcher
import candidate_parallel_sentence_pairs_classifier
import configparser
import sys

config = configparser.ConfigParser()
config.read('config.ini')

sys.path.insert(0, '../../common/')
import common

class ParallelSentencePairsIdentifier(object):
    def __init__(self):
        pass

    @staticmethod
    def result_analysis(predictions_path, gold_standard_path, en_path, zh_path, write_false_positive=False):
        correct_predictions_count = 0
        predictions_count = 0
        gold_standard_dict = {}

        # TODO path go to config
        if write_false_positive:
            en_file = common.read_two_columns_file_to_build_dictionary_type_specified(en_path, str, str)
            zh_file = common.read_two_columns_file_to_build_dictionary_type_specified(zh_path, str, str)
            count_source_in_gold = 0
            count_source_not_in_gold = 0

        with open(gold_standard_path) as f1:
            for line1 in f1:
                (target_id, source_id) = line1.rstrip('\n').split("\t")
                gold_standard_dict[source_id] = target_id

        with open(predictions_path) as f2:
            false_positive = open(config['output_files_for_test_data']['false_positive'], 'w')
            for line2 in f2:
                predictions_count += 1
                (target_id_predicted, source_id_predicted) = line2.rstrip('\n').split("\t")
                if source_id_predicted in gold_standard_dict:
                    if gold_standard_dict[source_id_predicted] == target_id_predicted:
                        correct_predictions_count += 1
                    elif write_false_positive:
                        false_positive.write('ingold\n' + source_id_predicted+ '\n' +en_file[source_id_predicted] + '\n'
                                             +target_id_predicted+'\n'+ zh_file[target_id_predicted] + '\n\n')
                        count_source_in_gold += 1
                elif write_false_positive:
                    false_positive.write(source_id_predicted+ '\n' +en_file[source_id_predicted] + '\n'
                                             +target_id_predicted+'\n'+ zh_file[target_id_predicted] + '\n\n')
                    count_source_not_in_gold += 1

        if write_false_positive:
            false_positive.write('count_source_in_gold' + str(count_source_in_gold))
            false_positive.write('count_source_not_in_gold' + str(count_source_not_in_gold))

        print('#real translations predicted (TP)', correct_predictions_count)
        print('#real translations (TP+FN)', len(gold_standard_dict))
        recall = correct_predictions_count / len(gold_standard_dict)
        print('recall (TP/(TP+FN))', recall)
        print('#translations predicted (TP+FP)', predictions_count)
        precision = correct_predictions_count / predictions_count
        print('precision (TP/(TP+FP))', precision)
        print('f1', 2 * (precision * recall) / (precision + recall))

    @staticmethod
    def intersection(dict1_path, dict2_path, pred_path):
        dict1 = dict()
        with open(dict1_path) as f:
            for line in f:
                (target_id, source_id) = line.rstrip('\n').split("\t")
                if target_id not in dict1:
                    dict1[target_id] = [source_id]
                else:
                    dict1[target_id].append(source_id)

        dict2 = dict()
        with open(dict2_path) as f:
            for line in f:
                (target_id, source_id) = line.rstrip('\n').split("\t")
                if target_id not in dict2:
                    dict2[target_id] = [source_id]
                else:
                    dict2[target_id].append(source_id)

        keys_a = set(dict1.keys())
        keys_b = set(dict2.keys())
        target_id_intersection = keys_a & keys_b
        print(target_id)

        pred = open(pred_path, 'w')
        for target_id in target_id_intersection:
            print(target_id)
            count = 1
            for source_id in dict1[target_id]:
                if source_id in dict2[target_id]:
                    pred.write(target_id + '\t' + source_id + '\n')
                    count += 1
                    if count == 3:
                        print('*************')

    @staticmethod
    def all():
        # # ------------------------------------- Translating ----------------------------------------------------------
        # print('\033[94m[translating training target file]\033[0m')
        # cct_train = chinese_corpus_translator.ChineseCorpusTranslator(
        #     zh_en_dict_path=config['dictionary']['zh_en_dict_for_training_data'],
        #     remove_chinese_stopwords=config.getboolean("preprocessing_text", "remove_chinese_stopwords"),
        #     english_remove_stopwords=config.getboolean("preprocessing_text", "remove_english_stopwords"),
        #     english_stem=config.getboolean("preprocessing_text", "english_stemming"),
        #     english_stem_for_dict=config.getboolean("preprocessing_text", "zh_en_dict_stemming"),
        #     chinese_tokenizer_mode_for_solr=config['preprocessing_text']['chinese_tokenizer_mode_for_solr'],
        #     chinese_tokenizer_mode_for_overlap=config['preprocessing_text']['chinese_tokenizer_mode_for_overlap'])
        # cct_train.translate(
        #     corpus_file_path=config['training_data']['zh'],
        #     unknown_words_path=config['output_files_for_training_data']['unknown_words'],
        #     translated_corpus_path=config['output_files_for_training_data']['translated_corpus_path'],
        #     translated_corpus_for_selecter_path=
        #     config['output_files_for_training_data']['translated_corpus_for_selecter_path'],
        #     translated_corpus_for_overlap_path=
        #     config['output_files_for_training_data']['translated_corpus_for_overlap_path'])
        #
        # print('\033[94m[translating test target file]\033[0m')
        # cct_test = chinese_corpus_translator.ChineseCorpusTranslator(
        #     zh_en_dict_path=config['dictionary']['zh_en_dict_for_test_data'],
        #     remove_chinese_stopwords=config.getboolean("preprocessing_text", "remove_chinese_stopwords"),
        #     english_remove_stopwords=config.getboolean("preprocessing_text", "remove_english_stopwords"),
        #     english_stem=config.getboolean("preprocessing_text", "english_stemming"),
        #     english_stem_for_dict=config.getboolean("preprocessing_text", "zh_en_dict_stemming"),
        #     chinese_tokenizer_mode_for_solr=config['preprocessing_text']['chinese_tokenizer_mode_for_solr'],
        #     chinese_tokenizer_mode_for_overlap=config['preprocessing_text']['chinese_tokenizer_mode_for_overlap'])
        # cct_test.translate(
        #     corpus_file_path=config['test_data']['zh'],
        #     unknown_words_path=config['output_files_for_test_data']['unknown_words'],
        #     translated_corpus_path=config['output_files_for_test_data']['translated_corpus_path'],
        #     translated_corpus_for_selecter_path=
        #     config['output_files_for_test_data']['translated_corpus_for_selecter_path'],
        #     translated_corpus_for_overlap_path=
        #     config['output_files_for_test_data']['translated_corpus_for_overlap_path'])

        # # ------------------------------------- Searching --------------------------------------------------------------
        # print('\033[94m[Solr searching for training data]\033[0m')
        # cpsps_train = candidate_parallel_sentence_pairs_searcher.CandidateParallelSentencePairsFinder(
        #     index_file_path=config['output_files_for_training_data']['translated_corpus_path'],
        #     index_file_for_solr_path=config['output_files_for_training_data']['corpus_for_solr'],
        #     english_remove_stopwords=config.getboolean("preprocessing_text", "remove_english_stopwords_in_source"),
        #     english_stem=config.getboolean("preprocessing_text", "english_stemming_in_source"))
        # cpsps_train.search_corpus(
        #     searching_file_path=config['training_data']['en'],
        #     output_path=config['output_files_for_training_data']['source_target_and_potential_targets_path'],
        #     gold_standard_file_path=config['training_data']['gold'])
        #
        # # # Reverse searching test
        # # print('\033[94m[Solr searching for training data]\033[0m')
        # # cpsps_train = candidate_parallel_sentence_pairs_searcher.CandidateParallelSentencePairsFinder(
        # #     index_file_path=config['training_data']['en'],
        # #     index_file_for_solr_path=config['output_files_for_training_data']['corpus_for_solr'])
        # # cpsps_train.search_corpus(
        # #     searching_file_path=config['output_files_for_training_data']['translated_corpus_path'],
        # #     output_path=config['output_files_for_training_data']['source_target_and_potential_targets_path'],
        # #     gold_standard_file_path=config['training_data']['gold'])
        #
        # print('\033[94m[Solr searching for test data]\033[0m')
        # cpsps_test = candidate_parallel_sentence_pairs_searcher.CandidateParallelSentencePairsFinder(
        #     index_file_path=config['output_files_for_test_data']['translated_corpus_path'],
        #     index_file_for_solr_path=config['output_files_for_test_data']['corpus_for_solr'],
        #     english_remove_stopwords=config.getboolean("preprocessing_text", "remove_english_stopwords_in_source"),
        #     english_stem=config.getboolean("preprocessing_text", "english_stemming_in_source"))
        # cpsps_test.search_corpus(
        #     searching_file_path=config['test_data']['en'],
        #     output_path=config['output_files_for_test_data']['source_target_and_potential_targets_path'])

        # ------------------------------------- Classifying ------------------------------------------------------------
        # print('\033[94m[SVM training]\033[0m')
        # cpspc_train = candidate_parallel_sentence_pairs_classifier.CandidateParallelSentencePairsClassifier()
        # cpspc_train.preprocessing_data(
        #     source_target_and_potential_targets_path=
        #     config['output_files_for_training_data']['source_target_and_potential_targets_path'],
        #     translated_target_information_path=
        #     config['output_files_for_training_data']['translated_corpus_for_selecter_path'],
        #     translated_corpus_for_overlap_path=
        #     config['output_files_for_training_data']['translated_corpus_for_overlap_path'],
        #     source_information_path=config['training_data']['en'],
        #     output_folder_path_prefix=config['output_files_for_training_data']['features_labels'])

        print('\033[94m[SVM predicting]\033[0m')
        cpspc_test = candidate_parallel_sentence_pairs_classifier.CandidateParallelSentencePairsClassifier()
        cpspc_test.prediction(
            training_folder_path=config['output_files_for_training_data']['features_labels'],
            test_source_target_and_potential_targets_path=
            config['output_files_for_test_data']['source_target_and_potential_targets_path'],
            test_translated_target_information_path=
            config['output_files_for_test_data']['translated_corpus_for_selecter_path'],
            test_translated_corpus_for_overlap_path=
            config['output_files_for_test_data']['translated_corpus_for_overlap_path'],
            test_source_information_path=config['test_data']['en'],
            test_output_folder_path_prefix=config['output_files_for_test_data']['features_labels']
        )


# # Whole system test (Test run 1)
# ParallelSentencePairsIdentifier.all()
ParallelSentencePairsIdentifier.result_analysis(
    predictions_path=config['output_files_for_test_data']['predictions'],
    gold_standard_path=config['test_data']['gold'],
    en_path=config['test_data']['en'],
    zh_path=config['test_data']['zh'],
    write_false_positive=True)

# # Test run 2
# ParallelSentencePairsIdentifier.result_analysis(
#     predictions_path='../data/predictions_config2_test',
#     gold_standard_path='../data/bucc2017/test_data/zh-en.test.gold')

# # Test run 3
# ParallelSentencePairsIdentifier.intersection(
#     dict1_path='../data/predictions',
#     dict2_path='../data/predictions_config2_test_10000',
#     pred_path='../data/predictions_config3_test'
# )
# ParallelSentencePairsIdentifier.result_analysis(
#     predictions_path='../data/predictions_config3_test',
#     gold_standard_path='../data/bucc2017/test_data/zh-en.test.gold')


# ParallelSentencePairsIdentifier.result_analysis(
#     predictions_path='../data/predictions_intersection',
#     gold_standard_path='../data/bucc2017/training_data/zh-en.training.gold')
