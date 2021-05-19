import re
import math
import nltk
from nltk.corpus import stopwords


def sentenceSegmenter(data):
    sentences = []
    if not isinstance(data, str):
        print("Error: input data is not string!")
        return sentences

    # find the character segment like
    # (dot*1)(whitespace*n)(Uppercase letter*1)
    sentences = re.split('(?<!Mr)(?<!Mrs)[?|.|!]\s*(?=[A-Z"])*', data)

    return sentences


"""
divide all the sentence to words
"""


def wordDivider(sentences):
    words = []

    sentences = [re.split('(?<!Mr.)(?<!Mrs.)[\s|\-|,|:|;|#|"]', sentence) for sentence in sentences]

    # remove 0-length strings
    for sentence in sentences:
        word_list = [word for word in sentence if len(word) > 0]
        if len(word_list) > 0:
            words += word_list

    return words


def wordNormalization(wordSentences):
    normalizedWords = []

    porter = nltk.PorterStemmer()
    normalizedWords = [porter.stem(t) for t in wordSentences]

    return normalizedWords


def remove_stop_words(word_list):
    stop_words = set(stopwords.words('english'))
    no_stop_word_list = [w for w in word_list if not w in stop_words]
    return no_stop_word_list


def frequencyAnalysis(normalizedWords):
    words = []
    for word in normalizedWords:
        words.append(word)

    freq_list = nltk.FreqDist(words)

    return freq_list


def normalize_a_news(news):
    sentence = sentenceSegmenter(news)
    words = wordDivider(sentence)
    word_normalised = wordNormalization(words)
    remove_stop_word = remove_stop_words(word_normalised)
    frequency_list = frequencyAnalysis(remove_stop_word)

    return frequency_list


def training_test_split(all_data, percent_of_training_data):
    num_of_training_data = round(len(all_data) * percent_of_training_data)
    train = all_data[:num_of_training_data]
    test = all_data[num_of_training_data:]

    return train, test


def add_new_elments(vocab, new_element_dict):
    for key, value in new_element_dict.items():
        if key in vocab:
            vocab[key] += value
        else:
            vocab[key] = value
    return vocab


def reduce_less_common_key_dict(vocab, times):
    reduced_dict = {}
    for k, v in vocab.items():
        if v >= times:
            reduced_dict[k] = v
    return reduced_dict


def prior_preb(vocab, vocab_pos, vocab_neg, alpha):
    prior_pos_dict = {}
    prior_neg_dict = {}
    for word, freq in vocab.items():
        if word in vocab_pos:
            prior_pos_dict[word] = (vocab_pos[word] + alpha) / (sum(vocab_pos.values()) + alpha * len(vocab_pos))
        else:
            prior_pos_dict[word] = alpha / (sum(vocab_pos.values()) + alpha * len(vocab_pos))

        if word in vocab_neg:
            prior_neg_dict[word] = (vocab_neg[word] + alpha) / (sum(vocab_neg.values()) + alpha * len(vocab_neg))
        else:
            prior_neg_dict[word] = alpha / (sum(vocab_neg.values()) + alpha * len(vocab_neg))
    return prior_pos_dict, prior_neg_dict


def cond_prob(item, cond_prior_prob, alpha):
    if item in cond_prior_prob:
        return cond_prior_prob[item]
    else:
        # the item not exist in the prob dict
        return alpha / (sum(cond_prior_prob.values()) + alpha * len(cond_prior_prob))


def prediction(test_X, covid_19_classifier):
    pred = []
    res = []
    for test_news in test_X:
        score_pos = math.log(covid_19_classifier['p_covid'])
        for title_word in test_news['title']:
            score_pos += math.log(
                cond_prob(title_word, covid_19_classifier['word_pos_prior'], covid_19_classifier['alpha']))
        for title_word in test_news['description']:
            score_pos += math.log(
                cond_prob(title_word, covid_19_classifier['word_pos_prior'], covid_19_classifier['alpha']))
        score_neg = math.log(covid_19_classifier['p_not_covid'])
        for title_word in test_news['title']:
            score_neg += math.log(
                cond_prob(title_word, covid_19_classifier['word_neg_prior'], covid_19_classifier['alpha']))
        for title_word in test_news['description']:
            score_neg += math.log(
                cond_prob(title_word, covid_19_classifier['word_neg_prior'], covid_19_classifier['alpha']))
        score = True if score_pos > score_neg else False
        pred.append(score)
        res.append(test_news['is_covid'] == score)
    return pred, res


"""
# calculate precision and recall
# TP: gd is true, predict true
# FN: gd is true, predict false
# FP: gd is false, predict true
# TN: gd is false, predict false
"""


def precision_recall(pred, ground_truth):
    TP, FN, FP, TN = 0, 0, 0, 0
    if len(pred) != len(ground_truth):
        print("prediction and ground truth do not match!")
        return None
    for i in range(0, len(ground_truth)):
        if ground_truth[i]:
            if pred[i]:
                TP += 1
            else:
                FN += 1
        else:
            if pred[i]:
                FP += 1
            else:
                TN += 1
    return TP, FN, FP, TN
