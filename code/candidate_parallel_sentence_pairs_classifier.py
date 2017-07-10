import numpy as np
import nltk
import csv
import pickle
import random
import re
from random import randint
import ast
from sklearn import svm
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import KFold
from sklearn.metrics import f1_score
from sklearn.metrics import precision_score
from sklearn.metrics import recall_score
from sklearn.metrics import confusion_matrix
from sklearn import preprocessing
from sklearn.model_selection import StratifiedShuffleSplit
from sklearn.model_selection import GridSearchCV
from sklearn.svm import SVC
from scipy.spatial.distance import cosine
from sklearn.externals import joblib
import pandas as pd
import visual_diagnostics
import configparser

config = configparser.ConfigParser()
config.read('config.ini')


class CandidateParallelSentencePairsClassifier(object):
    def __init__(self):
        pass

    @staticmethod
    def preprocessing_data(source_target_and_potential_targets_path,
                           translated_target_information_path,
                           translated_corpus_for_overlap_path,
                           source_information_path,
                           output_folder_path_prefix,
                           use_extra_positive_information=True,
                           show_detail=True,
                           ten_percent_sampling=False):

        def get_english_sentence_length(sentence):
            # INPUT:    this's [is] a(1) test!
            # OUTPUT:   ["this's", '[', 'is', ']', 'a', '(', '1', ')', 'test', '!']
            # tokens = re.findall(r"[\w']+|[.,!?;(){}[\]]", sentence)
            # split at ANY punctuation
            # OUTPUT:   ['this', "'", 's', '[', 'is', ']', 'a', '(', '1', ')', 'test', '!']
            tokens = re.findall(r"[\w]+|[^\s\w]", sentence)
            return len(tokens)

        def get_overlap(source_sentence, target_sentence_list):
            # TODO NOW for the word ends with , or other punctuation marks?
            source_sentence = source_sentence.lower().split(" ")
            if config.getboolean("preprocessing_text", "remove_english_stopwords_for_overlap"):
                source_sentence = [token for token in source_sentence if token not in stpwds]
            if config.getboolean("preprocessing_text", "english_stemming_for_overlap"):
                source_sentence = [stemmer.stem(token) for token in source_sentence]

            target_sentence_list_copy = []
            for token_translations_words_list in target_sentence_list:
                token_translations_words_list = token_translations_words_list.lower().split(" ")
                if config.getboolean("preprocessing_text", "remove_english_stopwords_for_overlap"):
                    token_translations_words_list = \
                        [token for token in token_translations_words_list if token not in stpwds]
                if config.getboolean("preprocessing_text", "english_stemming_for_overlap"):
                    token_translations_words_list = \
                        [stemmer.stem(token) for token in token_translations_words_list]
                target_sentence_list_copy.append(token_translations_words_list)

            count = 0
            for source_token in source_sentence:
                for token_translations_words_list in target_sentence_list_copy:
                    if source_token in token_translations_words_list:
                        count += 1
                        target_sentence_list_copy.remove(token_translations_words_list)
                        break

            len_valid_source_tokens = len(source_sentence)
            # TODO NOW if after stem or removing stopwords, token_translations is empty
            # TODO NOW => len_valid_target_tokens should reduce 1
            len_valid_target_tokens = len(target_sentence_list)
            result = count / (len_valid_source_tokens + len_valid_target_tokens - count)
            return result

        # nltk.data.path = ['/vol/datailes/tools/nlp/nltk_data/2016']
        nltk.data.path.append("/Users/zzcoolj/Code/NLTK/nltk_data")
        stpwds = set(nltk.corpus.stopwords.words("english"))
        stemmer = nltk.stem.PorterStemmer()

        # ---------------------------- Generate training set by using source_target_and_potential_targets --------------
        ''' 
        training_set structure:
            [source_id,      target_id,    Y/N, Solr searching result index]
        e.g.[['en-000035794', 'zh-000074056', 1, 1]]
        '''
        training_set = []
        with open(source_target_and_potential_targets_path, "r") as f:
            reader = csv.reader(f, delimiter='\t')
            source_target_potential_targets = list(reader)

        gold_standard_found = False
        for line in source_target_potential_targets:
            source_id = line[0]
            target_id = line[1]
            potential_targets = line[2:]
            # Take into count only topN Solr results for each source. topN should be smaller than maximum_rows.
            potential_targets = potential_targets[:int(config['svm_parameters']['solr_topN'])]
            # Take into count only Solr results which score is bigger than solr_score_threshold for each source.
            if float(config['svm_parameters']['solr_score_threshold']) != 0.0:
                potential_targets = [e for e in potential_targets if float(e.split('/')[1]) >
                                     float(config['svm_parameters']['solr_score_threshold'])]
                # It's possible that there's no potential targets left after selecting on score threshold.
                if not potential_targets:
                    continue

            i = 1
            if target_id == 'NONE':
                # Data generated by source file.
                for potential_target in potential_targets:
                    [potential_target_id, potential_target_score] = potential_target.split('/')
                    training_set.append([source_id, potential_target_id, 0, i, potential_target_score])
                    i += 1
            else:
                # Data generated by gold standard.
                gold_standard_found = True
                for potential_target in potential_targets:
                    [potential_target_id, potential_target_score] = potential_target.split('/')
                    if potential_target_id == target_id:
                        training_set.append([source_id, potential_target_id, 1, i, potential_target_score])
                    else:
                        training_set.append([source_id, potential_target_id, 0, i, potential_target_score])
                    i += 1

                if use_extra_positive_information:
                    potential_targets_ids = [e.split('/')[0] for e in potential_targets]
                    if target_id not in potential_targets_ids:
                        '''
                        If correct target is not in potential targets, this source_id target_id pair servers as an extra 
                        positive example for training. 
                        But we don't know Solr index and score of this target_id, we should make sure what we set wouldn't 
                        influence training.
                        '''
                        # TODO LATER random or average
                        # Solr searching result index is a random number between 1 and potential_targets_len (included)
                        given_solr_index = randint(1, len(potential_targets_ids))
                        # Solr score is the same score as random selected potential target
                        given_solr_score = potential_targets[given_solr_index - 1].split('/')[1]
                        training_set.append([source_id, target_id, 1, given_solr_index, given_solr_score])

        # ------------------------------- Generate node info by source (English) file & translated target file ---------

        with open(translated_target_information_path, "r") as f:
            reader = csv.reader(f)
            target_original_info = list(reader)

        source_original_info = []
        '''
        ATTENTION: sentence with " mark could be very dangerous for csv reader. So code below doesn't use csv reader.
        e.g.
        original csv file:
        en-000000567	"Participatory communication is the theory and practices of communication used to involve people in the decision-making of the development process.
        en-000000568	Therefore, the purpose of communication should be to make something common, or to share...meanings, perceptions, worldviews or knowledge.
        csv reader takes 
        "Participatory communication is the theory and practices of communication used to involve people in the decision-making of the development process.
        en-000000568	Therefore, the purpose of communication should be to make something common, or to share...meanings, perceptions, worldviews or knowledge.
        as one sentence, it ignores \n somehow.
        '''
        with open(source_information_path, "r") as f:
            for line in f:
                source_original_info.append(line.rstrip('\n').split("\t"))

        if ten_percent_sampling:
            # Use only 10% of training data (for testing)
            # Randomly get testing set from training set.
            to_keep = random.sample(range(len(training_set)), k=int(round(len(training_set) * 0.3)))
            training_set = [training_set[i] for i in to_keep]
            valid_ids_source = set()
            valid_ids_target = set()
            for element in training_set:
                valid_ids_source.add(element[0])
                valid_ids_target.add(element[1])
            tmp_target = [element for element in target_original_info if element[0] in valid_ids_target]
            tmp_source = [element for element in source_original_info if element[0] in valid_ids_source]
            target_original_info = tmp_target
            source_original_info = tmp_source
            del tmp_target
            del tmp_source

        node_info = target_original_info + source_original_info

        print('training set size:', len(training_set))

        # ---------------------------------- Go through training set and extract training features ---------------------

        # Features:
        '''
        tf-idf & LSA
        https://www.quora.com/Whats-the-difference-between-SVD-and-TF-IDF
        TFIDF is an alternative to bag-of-words representation. 
        Both Tf-Idf and Bag of Words are ways to make document vectors of dimension 1Xv (say j) . 
        v here is vocabulary size (around 150000 for English) . 
        LSA/SVD is a way to get a new vector j' from vectors like and including j , such that j' has dimensions 1Xu , 
        such that u<<v , but stil j' preserves the semantic qualities of j.
        This j' is way easier to calculate distances upon.
        '''
        overlap = []
        tfidf_cos = []
        # Solr_index = []
        Solr_score = []
        sentence_length_rate = []
        LSA = []

        # IDs is just a temporary variable that helps building ID_pos
        IDs = []
        ID_pos = {}
        for element in node_info:
            ID_pos[element[0]] = len(IDs)
            IDs.append(element[0])
        corpus = [element[1] for element in node_info]
        '''
        max_df is used for removing terms that appear too frequently, also known as "corpus-specific stop words".
        e.g. max_df = 0.50 means "ignore terms that appear in more than 50% of the documents".
        '''
        vectorizer = TfidfVectorizer(stop_words='english', max_df=float(config['svm_parameters']['max_df']),
                                     min_df=int(config['svm_parameters']['min_df']))
        '''
        Method fit_transform is a shortening for
        vect.fit(corpus)
        corpus_tf_idf = vect.transform(corpus)
        Learn vocabulary and idf, return term-document matrix.
        '''
        M = vectorizer.fit_transform(corpus)

        # TODO LSA memory error
        # print("Computing LSA")
        # # compute TFIDF vector of each paper
        # # LSA --keep part of the documents for memory reasons around 2k rows
        # corpus2 = [corpus[i] for i in sorted(random.sample(range(len(corpus)), k=1000))]
        # A = vectorizer.transform(corpus2).toarray()
        # # apply singular value decomposition (SVD)
        # U, S, V = np.linalg.svd(A)
        # # each row is a node in the order of target_original_info
        # # TODO features_TFIDF is exactly same as M_target, so replace it ?
        # features_TFIDF = vectorizer.fit_transform(corpus)
        # # keep the first 4 rows of v
        # V2 = V[:100, :]
        # M2 = np.dot(features_TFIDF.toarray(), V2.transpose())
        # print("LSA DONE")

        # Expand node_info information
        for node_info_element in node_info:
            '''
            node_info element structure:
                source(English):            [source_id, sentence]
                translated target(Chinese): [target_id, sentence, sentence_length]
            After processing:
                                            [target_id, sentence, sentence_length]
            '''
            # Source (English) sentence length
            if len(node_info_element) == 2:
                node_info_element.append(get_english_sentence_length(node_info_element[1]))

        translated_target_info_list_of_list = joblib.load(translated_corpus_for_overlap_path)

        counter = 0
        for i in range(len(training_set)):
            source_id = training_set[i][0]
            target_id = training_set[i][1]
            source_info = node_info[ID_pos[source_id]]
            target_info = node_info[ID_pos[target_id]]

            # sentence_length_rate
            len_source = int(source_info[2])
            len_target = int(target_info[2])
            sentence_length_rate.append(len_target / len_source)

            # overlap
            overlap.append(get_overlap(
                source_sentence=source_info[1],
                target_sentence_list=translated_target_info_list_of_list[target_id]))

            # tfidf_cos
            vector1 = M[ID_pos[source_id], :].toarray()[0]
            vector2 = M[ID_pos[target_id], :].toarray()[0]
            temp_cosine = 0.0
            if np.linalg.norm(vector1) != 0 and np.linalg.norm(vector2) != 0:
                temp_cosine = cosine(vector1, vector2)
            tfidf_cos.append(temp_cosine)

            # TODO LSA memory error
            # LSA.append(cosine(M2[ID_pos[target], :], M2[ID_pos[source], :]))

            # Solr_index.append(int(training_set[i][3]))
            Solr_score.append(float(training_set[i][4]))

            if show_detail:
                if counter % 10000 == 0:
                    print(str(counter), "training examples processed")
                counter += 1

        # convert list of lists into array
        # documents as rows, unique words as columns (i.e., example as rows, features as columns)
        # TODO LSA memory error
        training_features_original = np.array(
            [overlap, tfidf_cos, Solr_score, sentence_length_rate],
            dtype=np.float64).T
        training_features_scaled = preprocessing.scale(training_features_original)

        # Write training_features (original & scaled)
        labels_array_output_path = output_folder_path_prefix + 'labels.pkl'
        training_features_scaled_output_path = output_folder_path_prefix + 'features.pkl'
        training_features_original_output_path = output_folder_path_prefix + 'features_original.pkl'
        training_features_original_output = open(training_features_original_output_path, 'wb')
        pickle.dump(training_features_original, training_features_original_output)
        training_features_scaled_output = open(training_features_scaled_output_path, 'wb')
        pickle.dump(training_features_scaled, training_features_scaled_output)

        # If this function is used to deal with training data (gold_standard_found is True), save labels.
        if gold_standard_found:
            labels = [int(element[2]) for element in training_set]
            labels = list(labels)
            labels_array = np.array(labels)
            labels_array_output = open(labels_array_output_path, 'wb')
            pickle.dump(labels_array, labels_array_output)

        return training_features_scaled, training_set

    @staticmethod
    def evaluation(folder_path,
                   n_splits_num=5,
                   show_detail=True):
        print('\n------------------- Evaluation ---------------------')
        # Read data ----------------------------------------------------------------------------------------------------
        print("\033[94m[reading training features and labels]\033[0m")
        training_features_scaled_path = folder_path + 'training_features.pkl'
        labels_array_path = folder_path + 'labels_array.pkl'
        training_features_original_path = folder_path + 'training_features_original.pkl'

        pkl_file1 = open(training_features_scaled_path, 'rb')
        training_features_scaled = pickle.load(pkl_file1)
        pkl_file1.close()

        pkl_file2 = open(labels_array_path, 'rb')
        labels_array = pickle.load(pkl_file2)
        pkl_file2.close()

        # Evaluation ---------------------------------------------------------------------------------------------------
        print("\033[94m[evaluating]\033[0m")
        '''
        KFold divides all the samples in k groups of samples, 
        called folds (if k = n, this is equivalent to the Leave One Out strategy), of equal sizes (if possible). 
        The prediction function is learned using k - 1 folds, and the fold left out is used for test.
        '''
        kf = KFold(n_splits=n_splits_num, shuffle=True)
        sum_f1 = 0
        sum_precision = 0
        sum_recall = 0
        for train_index, test_index in kf.split(training_features_scaled):
            X_train, X_test = training_features_scaled[train_index], training_features_scaled[test_index]
            y_train, y_test = labels_array[train_index], labels_array[test_index]

            if show_detail:
                unique_train, counts_train = np.unique(y_train, return_counts=True)
                print('y_train-----------------------\n', np.asarray((unique_train, counts_train)).T)
                print('positive:negative => 1 :', counts_train[0] / counts_train[1])
                unique_test, counts_test = np.unique(y_test, return_counts=True)
                print('y_test------------------------\n', np.asarray((unique_test, counts_test)).T)
                print('positive:negative => 1 :', counts_test[0] / counts_test[1])

            # http://scikit-learn.org/stable/auto_examples/svm/plot_separating_hyperplane_unbalanced.html
            classifier = svm.SVC(kernel='rbf',
                                 class_weight=ast.literal_eval(config['svm_parameters']['class_weight']),
                                 C=float(config['svm_parameters']['C']),
                                 gamma=config['svm_parameters']['gamma'])
            classifier.fit(X_train, y_train)
            pred = classifier.predict(X_test)

            # pred = gsvm(X_train=X_train, y_train=y_train, X_test=X_test,
            #             majority_label=0, minority_label=1, classifier=svm.SVC(kernel='rbf'))
            # # print full array
            # np.set_printoptions(threshold=np.nan)
            # print(pred)
            # print('*******************************************')
            # print(y_test)

            f1_score_temp = f1_score(y_true=y_test, y_pred=pred)
            sum_f1 += f1_score_temp
            precision_score_temp = precision_score(y_true=y_test, y_pred=pred)
            sum_precision += precision_score_temp
            recall_score_temp = recall_score(y_true=y_test, y_pred=pred)
            sum_recall += recall_score_temp
            if show_detail:
                unique_pred, counts_pred = np.unique(pred, return_counts=True)
                print('pred--------------------------\n', np.asarray((unique_pred, counts_pred)).T)
                if len(counts_pred) == 2:
                    print('positive:negative => 1 :', counts_pred[0] / counts_pred[1], '\n')

                cm = confusion_matrix(y_true=y_test, y_pred=pred)
                print('#real translations predicted (TP)', cm[1][1])
                print('#real translations (TP+FN)', (cm[1][1] + cm[1][0]))
                print('recall (TP/(TP+FN))', recall_score_temp)
                print('#translations predicted (TP+FP)', cm[1][1] + cm[0][1])
                print('precision (TP/(TP+FP))', precision_score_temp)
                print('f1', f1_score_temp, '\n----------------------------------------------\n')
        print('f1_score', sum_f1 / n_splits_num)
        print('precision_score', sum_precision / n_splits_num)
        print('recall_score', sum_recall / n_splits_num, '\n')

        # data visualization -------------------------------------------------------------------------------------------
        print('\033[94m[features distribution visualization]\033[0m')
        pkl_file3 = open(training_features_original_path, 'rb')
        training_features_original = pickle.load(pkl_file3)
        pkl_file3.close()
        visual_diagnostics.hist_viz_for_two_classifications(
            features_array=training_features_original,
            labels_array=labels_array,
            class1_label=1,
            features_name=['overlap', 'tfidf_cos', 'Solr_index', 'Solr_score', 'sentence_length_rate'])

    @staticmethod
    def find_best_c_gamma(folder_path):
        print("\033[94m[reading training features and labels]\033[0m")
        training_features_scaled_path = folder_path + 'features.pkl'
        labels_array_path = folder_path + 'labels.pkl'

        pkl_file1 = open(training_features_scaled_path, 'rb')
        training_features_scaled = pickle.load(pkl_file1)
        pkl_file1.close()

        pkl_file2 = open(labels_array_path, 'rb')
        labels_array = pickle.load(pkl_file2)
        pkl_file2.close()

        C_range = np.logspace(0, 10, 1)
        gamma_range = np.logspace(-1, 3, 1)
        class_weight_range = np.array([{1: 8}, {1: 9}])
        param_grid = dict(gamma=gamma_range, C=C_range, class_weight=class_weight_range)
        '''
        Stratification is the process of rearranging the data as to ensure each fold is a good representative 
        of the whole. For example in a binary classification problem where each class comprises 50% of the data, 
        it is best to arrange the data such that in every fold, each class comprises around half the instances.
        '''
        cv = StratifiedShuffleSplit(n_splits=2, test_size=0.5, random_state=2)
        grid = GridSearchCV(SVC(kernel='rbf'), param_grid=param_grid, cv=cv, n_jobs=1, scoring='f1')
        grid.fit(training_features_scaled, labels_array)

        print(pd.DataFrame(grid.cv_results_))
        print("\nThe best parameters are %s with a score of %0.2f"
              % (grid.best_params_, grid.best_score_))

    # On the assumption that we already have training features and labels by using preprocessing_data function.
    # Attention: Make sure they use same parameters of preprocessing_data function.
    def prediction(self,
                   training_folder_path,
                   test_source_target_and_potential_targets_path,
                   test_translated_target_information_path,
                   test_translated_corpus_for_overlap_path,
                   test_source_information_path,
                   test_output_folder_path_prefix
                   ):

        """
        test_labels_array_output_path and use_extra_positive_information are useless for test data.
        The reason to keep them is just to use preprocessing_data function for both training data and test data.
        """
        print("[reading training features and labels]")
        training_features_scaled_path = training_folder_path + 'features.pkl'
        labels_array_path = training_folder_path + 'labels.pkl'
        pkl_file1 = open(training_features_scaled_path, 'rb')
        training_features_scaled = pickle.load(pkl_file1)
        pkl_file1.close()
        pkl_file2 = open(labels_array_path, 'rb')
        labels_array = pickle.load(pkl_file2)
        pkl_file2.close()

        print('[SVM model training]')
        classifier = svm.SVC(kernel='rbf',
                             class_weight=ast.literal_eval(config['svm_parameters']['class_weight']),
                             C=float(config['svm_parameters']['C']),
                             gamma=config['svm_parameters']['gamma'])
        classifier.fit(training_features_scaled, labels_array)
        joblib.dump(classifier, config['output_files_for_training_data']['trained_classifier'])

        print('[generating test features]')
        # There is no extra positive information for test data => use_extra_positive_information=False
        # Extract all information from test data =>  ten_percent_sampling=False
        test_features_scaled, test_training_set = self.preprocessing_data(
            source_target_and_potential_targets_path=test_source_target_and_potential_targets_path,
            translated_target_information_path=test_translated_target_information_path,
            translated_corpus_for_overlap_path=test_translated_corpus_for_overlap_path,
            source_information_path=test_source_information_path,
            output_folder_path_prefix=test_output_folder_path_prefix,
            use_extra_positive_information=False,
            ten_percent_sampling=False)
        joblib.dump(test_training_set, test_output_folder_path_prefix+'test_training_set.pkl')

        print('[SVM model predicting]')
        # classifier = joblib.load(config['output_files_for_training_data']['trained_classifier'])
        pred = classifier.predict(test_features_scaled)

        print("[generating csv file]")
        final_result = open(config['output_files_for_test_data']['predictions'], 'w')
        for index, predicted_label in enumerate(pred):
            if predicted_label == 1:
                final_result.write(test_training_set[index][1] + '\t' + test_training_set[index][0] + '\n')

# # For test
# CandidateParallelSentencePairsClassifier.evaluation(folder_path='../data/temp_data/classifier/training/')

# CandidateParallelSentencePairsClassifier.find_best_c_gamma(
#     folder_path='../data/temp_data/classifier/training_temp/')
