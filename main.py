# %%
import my_util

# %%
small_sample = "./SDU_Text_Mining_Project/small_sample/"
bbc_data_path = "./SDU_Text_Mining_Project/bbc_news_2020/"
nbc_data_path = "./SDU_Text_Mining_Project/nbc_news_2020/"
covid_19_classifier_file = "covid_19_classifier_bbc"
# %%
# read all the data using given data_path.
bbc_X, bbc_y = my_util.read_data(bbc_data_path)
# make a covid_19 classifier
covid_19_classifier = my_util.make_classifier(covid_19_classifier_file, bbc_X, bbc_y)

# %%
# Estimations
# - As proportion of all articles in 2020
print("\n(bbc 2020) Evaluation")
my_util.classifier_evaluation(covid_19_classifier, bbc_X, bbc_y)

# - As proportion of all articles in each month of 2020
monthly_bbc_news = my_util.get_all_folder_names(bbc_data_path)
for i in range(0, len(monthly_bbc_news)):
    print(f"(bbc month {monthly_bbc_news[i]}) Evaluation")
    bbc_month_X, bbc_month_y = my_util.read_data(bbc_data_path + monthly_bbc_news[i] + "/")
    my_util.classifier_evaluation(covid_19_classifier, bbc_month_X, bbc_month_y)

# - As proportion of articles in an outlet (e.g. CNN) in 2020
print("\n(nbc 2020) Evaluation")
nbc_X, nbc_y = my_util.read_data(nbc_data_path)
my_util.classifier_evaluation(covid_19_classifier, nbc_X, nbc_y)

# %%

entities = my_util.ner(bbc_data_path)
file_name = "entities_bbc.txt"
my_util.save_data_to_file(entities, file_name)

result = "./SDU_Text_Mining_Project/entities_bbc.txt"
entities = my_util.load_file_to_data(result)

most_common_elements = my_util.get_most_common_elements(entities, 10)
for each in most_common_elements:
    print(each)
# %%
print("done!")
