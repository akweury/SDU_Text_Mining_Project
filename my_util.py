import glob
import gzip
import json
import pickle
from collections import Counter
import os
from os import path
from os import listdir
from os.path import join

import textMiningUtil

"""
get all the files in a folder with determined suffix
"""


def get_all_files_in_folder(folder_path, suffix):
    file_paths = glob.glob(folder_path + suffix)
    return file_paths


"""
unzip a list of gz files to string lists 
"""


def unzip_gz_files(zip_file_list):
    unzipped_file_list = []
    for file in zip_file_list:
        f = gzip.open(file, 'rt')
        file_content = json.load(f)
        unzipped_file_list.append(file_content)
        f.close()
    return unzipped_file_list


"""
save data to the file "file_name"
"""


def save_data_to_file(data, file_name):
    with open(file_name, 'wb') as fp:
        pickle.dump(data, fp, protocol=pickle.HIGHEST_PROTOCOL)


def save_data_to_txt_file(data, file_name):
    with open(file_name, 'wt') as fp:
        pickle.dump(data, fp, protocol=pickle.HIGHEST_PROTOCOL)


"""
given a file_name, load it to a variable
"""


def load_file_to_data(file_name):
    with open(file_name, 'rb') as fp:
        data = pickle.load(fp)
    return data


def load_txt(file):
    with open(file, 'rt') as fp:
        data = fp.readlines()
    return data


"""
check if file_name exists
"""


def has_file(file_name):
    return path.exists(file_name)


"""
load all the files with specified suffix
"""


def load_dataset(dataset, suffix):
    data = get_all_files_in_folder(dataset, suffix)

    data = unzip_gz_files(data)
    all_news = []
    for all_day_news in data:
        all_news += all_day_news.values()
    print(f"{len(all_news)} news are going to be analyzed")
    return all_news


def read_data(data_path):
    dataset_file = data_path + 'all_news.data'
    label_file = data_path + 'all_news.label'
    if has_file(dataset_file):
        print("dataset exists, loading...")
        X = load_file_to_data(dataset_file)
        y = load_file_to_data(label_file)
    else:
        print("dataset not exist, creating...")
        news_list = load_dataset(data_path, "*.gz")

        # create a normalization pipeline, use all the titles and descriptions
        X = []
        y = []
        for news in news_list:
            title = textMiningUtil.normalize_a_news(news['title'])
            description = textMiningUtil.normalize_a_news(news['description'])
            label = news['is_covid']
            dict = {'title': title,
                    'description': description,
                    'is_covid': label}
            X.append(dict)
            y.append(label)
        save_data_to_file(X, dataset_file)
        save_data_to_file(y, label_file)
    return X, y


def classifier_evaluation(classifier, test_dataset, test_label):
    # make the predictions to evaluate the classifier
    pred, res = textMiningUtil.prediction(test_dataset, classifier)
    accurate_rate = round(res.count(True) / len(res), 2)
    print(f"accurate rate: {accurate_rate}")

    # calculate TP, FP
    # TP: is covid, predict covid
    # FN: is covid, predict not covid
    # FP: not covid, predict covid
    # TN: not covid, predict not covid
    TP, FN, FP, TN = textMiningUtil.precision_recall(pred, test_label)
    accuracy = (TN + TP) / (TN + TP + FN + FP)
    print(f"accuracy: {accuracy} ")
    return accuracy


def make_classifier(classifier_file, X, y):
    if has_file(classifier_file):
        print("covid 19 classifier exists, loading...")
        covid_19_classifier = load_file_to_data(classifier_file)
        print(f"Covid-19 classifier\n "
              f"Data Resource: {covid_19_classifier['DataResource']}\n "
              f"Accuracy: {covid_19_classifier['accuracy']}")
        return covid_19_classifier
    else:
        # train classifier
        print("Covid 19 classifier not exists, start training...")
        # partition the data to the train set and test set
        train_X, test_X = textMiningUtil.training_test_split(X, 0.9)
        train_y, test_y = textMiningUtil.training_test_split(y, 0.9)
        vocab = {}  # get all the words appeared in the training dataset
        vocab_pos = {}
        vocab_neg = {}

        for train_news in train_X:
            vocab = textMiningUtil.add_new_elments(vocab, train_news['title'])
            vocab = textMiningUtil.add_new_elments(vocab, train_news['description'])
            if train_news['is_covid']:
                vocab_pos = textMiningUtil.add_new_elments(vocab_pos, train_news['title'])
                vocab_pos = textMiningUtil.add_new_elments(vocab_pos, train_news['description'])
            else:
                vocab_neg = textMiningUtil.add_new_elments(vocab_neg, train_news['title'])
                vocab_neg = textMiningUtil.add_new_elments(vocab_neg, train_news['description'])

        # get p_covid, p_not_covid
        p_covid = train_y.count(True) / len(train_y)
        p_not_covid = 1 - p_covid
        # delete the word which appearing less than 2 times
        vocab_reduced = textMiningUtil.reduce_less_common_key_dict(vocab, 3)
        vocab_neg_reduced = textMiningUtil.reduce_less_common_key_dict(vocab_neg, 3)
        vocab_pos_reduced = textMiningUtil.reduce_less_common_key_dict(vocab_pos, 3)

        # get prior, which is p(word | class) for each word
        alpha = 1
        word_pos_prior, word_neg_prior = textMiningUtil.prior_preb(vocab, vocab_pos, vocab_neg, alpha)
        word_pos_prior_reduced, word_neg_prior_reduced = textMiningUtil.prior_preb(vocab_reduced, vocab_pos_reduced,
                                                                                   vocab_neg_reduced, alpha)

        covid_19_classifier = {'p_covid': p_covid,
                               'p_not_covid': p_not_covid,
                               'word_pos_prior': word_pos_prior,
                               'word_neg_prior': word_neg_prior,
                               'word_pos_prior_reduced': word_pos_prior_reduced,
                               'word_neg_prior_reduced': word_neg_prior_reduced,
                               'alpha': alpha,
                               'DataResource': 'bbc.com'}

        # make the predictions to evaluate the classifier
        accuracy = classifier_evaluation(covid_19_classifier, test_X, test_y)
        covid_19_classifier['accuracy'] = accuracy

        # save the classifier
        save_data_to_file(covid_19_classifier, classifier_file)
        return covid_19_classifier


def get_all_folder_names(folder_path):
    """

    :param folder_path: get all the folder names under the 'folder_path'
    :return: an array including all the folder names under this folder
    """

    onlyfolders = [f for f in listdir(folder_path) if os.path.isdir(join(folder_path, f))]
    return onlyfolders


def get_all_the_articles(data_path):
    news_list = load_dataset(data_path, "*.gz")
    articles = []
    for news in news_list:
        # title = textMiningUtil.sentenceSegmenter(news['title'])
        description = textMiningUtil.sentenceSegmenter(news['description'])
        articles.append(description)
    return articles


def ner(data_path):
    articles = get_all_the_articles(data_path)
    print(f"{len(articles)} articles are going to be analyzed.")

    entities = textMiningUtil.analysis_article(articles)

    Counter(entities).most_common(3)

    return entities


def save_entities(entities, file_name):
    with open(file_name, 'wt') as f:
        for k, v in entities.most_common():
            f.write("{} {}\n".format(k, v))


def get_most_common_elements(my_list, n):
    return Counter(my_list).most_common(n)
