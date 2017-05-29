from urllib.request import urlopen
from urllib.parse import quote_plus
import urllib
import os
import configparser
import nltk
import string
import sys
import csv

config = configparser.ConfigParser()
config.read('config.ini')

sys.path.insert(0, '../../common/')
import common
import visual_diagnostics


class CandidateParallelSentencePairsFinder(object):
    def __init__(self, index_file_path, english_remove_stopwords=False, english_stem=True):
        # Remove previous server data (assuming that Solr has already been started)
        print('\033[94m[Solr loading data]\033[0m')
        urlopen(
            'http://localhost:8983/solr/gettingstarted/'
            'update?stream.body=<delete><query>*:*</query></delete>&commit=true')
        # Load data into Solr server
        index_file_absolute_path = os.getcwd().replace('code', index_file_path.replace('../', ''))
        os.system('cd /Users/zzcoolj/Code/bucc2017/solr-6.4.2/;'
                  'bin/post -c gettingstarted ' + index_file_absolute_path)
        self.english_remove_stopwords = english_remove_stopwords
        self.english_stem = english_stem
        if english_remove_stopwords:
            self.stpwds = set(nltk.corpus.stopwords.words("english"))
        if english_stem:
            self.stemmer = nltk.stem.PorterStemmer()

    def __pre_process_sentence(self, sentence):
        # Invalid characters
        result = sentence.replace('“', '"')
        result = result.replace('”', '"')

        if result.endswith('"'):
            result = result[:-1]
        '''
                                Space = > +
                                + = > % 2B
                                ( = > % 28
                                ) = > % 29
                                %5B is '['
                                %5D is ']'
                                / = > % 2F
                                ~ = > % 7E
                                } = > % 7D
                                ! = > % 21
                                + = > % 2B
                                | = > % 7C
                '''
        result = result.replace('"', ' ')
        result = result.replace(':', ' ')
        result = result.replace('(', ' ')
        result = result.replace(')', ' ')
        result = result.replace('[', ' ')
        result = result.replace(']', ' ')
        result = result.replace('/', ' ')
        result = result.replace('-', ' ')
        result = result.replace('{', ' ')
        result = result.replace('}', ' ')
        result = result.replace('~', ' ')
        result = result.replace('^', ' ')
        result = result.replace('!', ' ')
        result = result.replace('+', ' ')
        result = result.replace('|', ' ')

        if self.english_remove_stopwords or self.english_stem:
            result_list = result.lower().translate(str.maketrans('', '', string.punctuation)).split(" ")
            if self.english_remove_stopwords:
                result_list = [token for token in result_list if token not in self.stpwds]
            if self.english_stem:
                result_list = [self.stemmer.stem(token) for token in result_list]
            result = ' '.join(result_list)

        # TODO LATER q may be the key words of sentence

        # Format url query
        result = quote_plus(result)
        return result

    @staticmethod
    def __search_potential_target_sentences(sentence, rows):
        query = 'http://localhost:8983/solr/gettingstarted/select?' \
                + 'fl=id,score&' \
                  'indent=on' \
                  '&q=sentence:' \
                + sentence \
                + '&rows=' \
                + str(rows) \
                + '&wt=python'
        try:
            connection = urlopen(query)
            response = eval(connection.read())
            # print(response['response']['numFound'], "documents found.")
            results = [document['id'] + '/' + str(document['score']) for document in response['response']['docs']]
            return results
        except urllib.error.HTTPError as err:
            print(query)
            print(err.code)
            exit()
            return ['test']

    def search_corpus(self, searching_file_path, output_path,
                      gold_standard_file_path=None, only_search_gold_standard=False):
        source_target_and_potential_targets = dict()
        source_dict = common.read_two_columns_file_to_build_dictionary_type_specified(searching_file_path, str, str)

        counter = 0
        if gold_standard_file_path is not None:
            # Generate source_target_and_potential_targets by using gold standard.
            with open(gold_standard_file_path) as f:
                for line in f:
                    (target_id, source_id) = line.rstrip('\n').split("\t")
                    # TODO NOW searching_file is source file?
                    source_sentence = source_dict[source_id]
                    searching_sentence = self.__pre_process_sentence(source_sentence)
                    target_potential_targets_ids = self.__search_potential_target_sentences(searching_sentence,
                                                                                          config['solr_parameters'][
                                                                                              'rows'])
                    target_potential_targets_ids.insert(0, target_id)
                    source_target_and_potential_targets[source_id] = target_potential_targets_ids
                    counter += 1
                    if counter % 5000 == 0:
                        print(counter, "sentences searched")

        if not only_search_gold_standard:
            # Generate source_target_and_potential_targets by using source file.
            for source_id, source_sentence in source_dict.items():
                if source_id in source_target_and_potential_targets:
                    # Already added by gold standard
                    continue
                else:
                    searching_sentence = self.__pre_process_sentence(source_sentence)
                    target_potential_targets_ids = self.__search_potential_target_sentences(searching_sentence,
                                                                                          config['solr_parameters'][
                                                                                              'rows'])
                    target_potential_targets_ids.insert(0, 'NONE')
                    source_target_and_potential_targets[source_id] = target_potential_targets_ids
                    counter += 1
                    if counter % 5000 == 0:
                        print(counter, "sentences searched")

        # Write
        common.write_dict_to_file_value_type_specified(output_path, source_target_and_potential_targets, list)

    @staticmethod
    def evaluate_search_engine_result(source_target_and_potential_targets_path,
                                      gold_standard_num,
                                      find_best_rows_score_combination_parameters_decided,
                                      search_gold_standard, search_source_file):

        def get_how_many_gold_standard_left(score_threshold, rows):
            # TODO LATER Need more efficient algorithm
            # gold standard number
            true_num = 0
            # source-target pair number after selecting on rows and score threshold
            all_num = 0
            for element in target_and_potential_targets:
                target_id = element[0]
                potential_targets = element[1:rows + 1]
                potential_targets_ids = [e.split('/')[0] for e in potential_targets if
                                         float(e.split('/')[1]) > score_threshold]
                all_num += len(potential_targets_ids)
                if target_id == 'NONE':
                    continue
                if target_id in potential_targets_ids:
                    true_num += 1
            return true_num, all_num

        def show_solr_score_distribution(source_target_and_potential_targets_path, potential_targets_size):
            with open(source_target_and_potential_targets_path, "r") as file:
                csv_reader = csv.reader(file, delimiter='\t')
                target_and_potential_targets = list(csv_reader)
                target_and_potential_targets = [e[1:2 + potential_targets_size] for e in target_and_potential_targets]
            gold_standard_score = []
            false_potential_targets_score = []
            for element in target_and_potential_targets:
                target_id = element[0]
                potential_targets = element[1:]
                potential_targets_id = [e.split('/')[0] for e in potential_targets]
                potential_targets_score = [float(e.split('/')[1]) for e in potential_targets]
                potential_targets_score_copy = potential_targets_score[:]
                if target_id in potential_targets_id:
                    target_index = potential_targets_id.index(target_id)
                    score = potential_targets_score[target_index]
                    gold_standard_score.append(score)
                    del potential_targets_score_copy[target_index]
                elif target_id == 'NONE':
                    # This source sentence does not have translation (corresponding target sentence).
                    pass
                else:
                    # This source sentence has translation/target sentence.
                    # But the target sentence is not in the top maximum_rows Solr searching results.
                    # Set 5 as score for the targets which are not in potential targets.
                    gold_standard_score.append(-5)
                false_potential_targets_score.extend(potential_targets_score_copy)
            return gold_standard_score, false_potential_targets_score

        def find_best_rows_score_combination(success, score_start, score_step, rows_start):
            gold_standard_left_num_threshold = gold_standard_num * success
            rows_score_count = []
            # rows: rows_start, ..., 1
            for rows in range(rows_start, 0, -1):
                score_threshold_temp = score_start
                gold_standard_left_num, all_left_num = get_how_many_gold_standard_left(score_threshold_temp, rows)
                if gold_standard_left_num < gold_standard_left_num_threshold:
                    break
                score_threshold_log = score_threshold_temp
                all_left_num_log = all_left_num
                while gold_standard_left_num >= gold_standard_left_num_threshold:
                    score_threshold_log = score_threshold_temp
                    all_left_num_log = all_left_num
                    score_threshold_temp += score_step
                    gold_standard_left_num, all_left_num = get_how_many_gold_standard_left(score_threshold_temp, rows)
                rows_score_count.append([rows, score_threshold_log, all_left_num_log])
            rows_score_count.sort(key=lambda x: x[2])
            best_rows_found = int(rows_score_count[0][0])
            best_score_threshold_found = float(rows_score_count[0][1])
            return rows_score_count, best_rows_found, best_score_threshold_found

        if type(source_target_and_potential_targets_path) is dict:
            target_and_potential_targets = source_target_and_potential_targets_path.values()
            print('need path')
            exit()
            pass
        elif type(source_target_and_potential_targets_path) is str:
            with open(source_target_and_potential_targets_path, "r") as f:
                reader = csv.reader(f, delimiter='\t')
                target_and_potential_targets = list(reader)
                target_and_potential_targets = [e[1:] for e in target_and_potential_targets]
        else:
            raise ValueError('source_target_and_potential_targets_path type error')

        if search_gold_standard:
            if not find_best_rows_score_combination_parameters_decided:
                '''
                Code blocked below aims to helping user find good parameters for find_best_rows_score_combination 
                    function.
                If score_start starts from 0.0, it will take too much time for find_best_rows_score_combination 
                    calculation.
                If score_start is too big, there will be less (rows, score_threshold) pairs.
                '''
                # Make decision of score_start parameter by graph below (1)
                # Evaluate success on different score (with rows=maximum_rows_for_solr_searching)
                success_score_threshold_dict = dict()
                # Solr score threshold from 0 to 40
                for i in range(40):
                    true_count, useless = get_how_many_gold_standard_left(i, config['solr_parameters']['rows'])
                    precision = "{0:.5f}%".format(true_count / gold_standard_num * 100)
                    success_score_threshold_dict[str(i)] = precision
                # common.write_dict_to_file(precision_file_output_path + '_scoreThreshold', success_score_threshold_dict, 'str')
                # visual_diagnostics.plot_dict(precision_file_output_path + '_scoreThreshold')

                # Make decision of score_start parameter by graph below (2)
                if search_source_file:
                    # Show Solr score distribution
                    gold_scores, false_scores = show_solr_score_distribution(
                        source_target_and_potential_targets_path=source_target_and_potential_targets_path,
                        potential_targets_size=config['solr_parameters']['rows'])
                    visual_diagnostics.plot_two_lists(gold_scores, false_scores, bins=30)
            else:
                if search_source_file:
                    success = 0.80
                    best_list, best_rows_found, best_score_threshold_found = \
                        find_best_rows_score_combination(success=success, score_start=10.0, score_step=0.3,
                                                         rows_start=5)
                    print('To get the success of', success, 'for gold standard:')
                    print(best_list)
                    print('best rows found:', best_rows_found)
                    print('best score threshold found:', best_score_threshold_found)
                    # TODO LATER maybe this graph is not so useful. Show Solr score distribution
                    gold_scores, false_scores = show_solr_score_distribution(
                        source_target_and_potential_targets_path=source_target_and_potential_targets_path,
                        potential_targets_size=best_rows_found)
                    print('score distribution when rows=' + str(best_rows_found))
                    visual_diagnostics.plot_two_lists(gold_scores, false_scores, bins=30)

            # Evaluate success on different rows
            success_rows_dict = dict()
            for i in range(int(config['solr_parameters']['rows'])):
                true_count, useless = get_how_many_gold_standard_left(0.0, i + 1)
                # precision = "{0:.5f}%".format(true_count / gold_standard_num * 100)
                precision = true_count / gold_standard_num * 100
                success_rows_dict[str(i + 1)] = precision
            common.write_dict_to_file(config['output_files_for_training_data']['success'], success_rows_dict, 'str')
            visual_diagnostics.plot_dict(config['output_files_for_training_data']['success'])

            # Compare two Solr success rates
            # precision_accurate_mode_Stem_keepChineseStopWords_ccldc_processUnknownWords_keepASCII
            visual_diagnostics.plot_two_bar_side_by_side(
                config['output_files']['precision_file_output_path'],
                '/Users/zzcoolj/Code/bucc2017/data/temp_data/solr_success/'
                'precision_accurate_mode_Stem_keepChineseStopWords_ccldc_processUnknownWords_keepASCII')

        elif search_source_file:
            # Show Solr score distribution
            gold_scores, false_scores = show_solr_score_distribution(
                source_target_and_potential_targets_path=source_target_and_potential_targets_path,
                potential_targets_size=config['solr_parameters']['rows'])
            visual_diagnostics.plot_one_list(false_scores, bins=30)

    @staticmethod
    def get_results_by_score(source_target_and_potential_targets_path,
                             pred_path,
                             results_size, potential_targets_size):
        with open(source_target_and_potential_targets_path, "r") as f:
            reader = csv.reader(f, delimiter='\t')
            source_target_potential_targets = list(reader)

        result_set = []
        for line in source_target_potential_targets:
            source_id = line[0]
            potential_targets = line[2:potential_targets_size + 2]
            i = 1
            # Data generated by source file.
            for potential_target in potential_targets:
                [potential_target_id, potential_target_score] = potential_target.split('/')
                result_set.append([source_id, potential_target_id, i, float(potential_target_score)])
                i += 1
        result_set.sort(key=lambda x: x[3], reverse=True)
        result_set = result_set[:results_size]
        print(result_set)
        pred = open(pred_path, 'w')
        for e in result_set:
            pred.write(e[1] + '\t' + e[0] + '\n')

    @staticmethod
    def stop_solr():
        # Stop Solr server & remove server data
        os.system("cd /Users/zzcoolj/Code/bucc2017/solr-6.4.2/;"
                  "bin/solr stop -all;"
                  "rm -rf example/cloud/")


# # Corpus searching test
# cpspf = CandidateParallelSentencePairsFinder(index_file_path='/Users/zzcoolj/Code/bucc2017/data/temp_data/'
#                                                              'translated_target_file_test_full_mode_for_solr.csv')
# cpspf.search_corpus(source_file_path='/Users/zzcoolj/Code/bucc2017/data/bucc2017/training_data/zh-en.training.en',
#                     gold_standard_file_path='/Users/zzcoolj/Code/bucc2017/data/bucc2017/'
#                                             'training_data/zh-en.training.gold',
#                     only_search_gold_standard=True)

# # Second configuration: Replacing the classifier (Step 3) with a base- line ranking method based on the Solr score:
#         # we select the M sentences pairs with the highest scores,
#         # where M is determined according to the prior probability of being a correct sentence pair,
#         # estimated on the training data.
# CandidateParallelSentencePairsFinder.get_results_by_score(
#     source_target_and_potential_targets_path='../data/temp_data/source_target_and_potential_targets_test',
#     pred_path='../data/predictions_config2_test_10000',
#     results_size=10000,
#     potential_targets_size=1
# )

CandidateParallelSentencePairsFinder.evaluate_search_engine_result(
    source_target_and_potential_targets_path='../data/temp_data/source_target_and_potential_targets_training_temp',
    gold_standard_num=1899,
    find_best_rows_score_combination_parameters_decided=True,
    search_gold_standard=True,
    search_source_file=False
)
