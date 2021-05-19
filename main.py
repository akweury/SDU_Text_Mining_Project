# %%
import my_util
import textMiningUtil

# %%
# read all the file as json list
small_sample = "./SDU_Text_Mining_Project/small_sample/"
dataset = "./SDU_Text_Mining_Project/2020_news/"

# folder_path = small_sample
folder_path = dataset
suffix = "*.gz"
data = my_util.get_all_files_in_folder(folder_path, suffix)
data = my_util.unzip_gz_files(data)

# %%
all_news = []
for all_day_news in data:
    all_news += all_day_news.values()
print(f"{len(all_news)} news are going to be analyzed")


# %%
# create a normalization pipeline, use all the titles and descriptions
def normalize_a_news(news):
    sentence = textMiningUtil.sentenceSegmenter(news)
    words = textMiningUtil.wordDivider(sentence)
    word_normalised = textMiningUtil.wordNormalization(words)
    remove_stop_word = textMiningUtil.remove_stop_words(word_normalised)
    frequency_list = textMiningUtil.frequencyAnalysis(remove_stop_word)

    return frequency_list


# data processing
print("data extracting...")
dataset_file = dataset + 'all_news.data'
label_file = dataset + 'all_news.label'
if my_util.has_file(dataset_file):
    print("dataset exists, loading...")
    X = my_util.load_file_to_data(dataset_file)
    y = my_util.load_file_to_data(label_file)
else:
    print("dataset not exist, creating...")
    X = []
    y = []
    for news in all_news:
        title = normalize_a_news(news['title'])
        description = normalize_a_news(news['description'])
        label = news['is_covid']
        dict = {'title': title,
                'description': description,
                'is_covid': label}
        X.append(dict)
        y.append(label)
    my_util.save_data_to_file(X, dataset_file)
    my_util.save_data_to_file(y, label_file)

# %%
print("start training...")
# train classifier
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

# %%
# test classifier
# posterior is p(class | word)

pred, res = textMiningUtil.prediction(test_X, p_covid, p_not_covid,
                                      word_pos_prior, word_neg_prior, alpha)
pred_reduced, res_reduced = textMiningUtil.prediction(test_X, p_covid, p_not_covid,
                                                      word_pos_prior_reduced, word_neg_prior_reduced, alpha)

accurate_rate = round(res.count(True) / len(res), 2)
accurate_rate_reduced = round(res_reduced.count(True) / len(res_reduced), 2)
print(f"accurate rate: {accurate_rate}")
print(f"accurate rate (reduced): {accurate_rate_reduced}")

# %%
# calculate TP, FP
# TP: is covid, predict covid
# FN: is covid, predict not covid
# FP: not covid, predict covid
# TN: not covid, predict not covid
TP, FN, FP, TN = textMiningUtil.precision_recall(pred, test_y)
accuracy = (TN + TP) / (TN + TP + FN + FP)

# %%
print("done!")
