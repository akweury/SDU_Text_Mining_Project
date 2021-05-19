# %%
import my_util

# %%
small_sample = "./SDU_Text_Mining_Project/small_sample/"
bbc_data_path = "./SDU_Text_Mining_Project/2020_news/"
cnn_data_path = ""
covid_19_classifier_file = "covid_19_classifier_bbc"
# %%
# read all the data using given data_path.
X, y = my_util.read_data(bbc_data_path)

# %%
# make a covid_19 classifier
covid_19_classifier = my_util.make_classifier(covid_19_classifier_file, X, y)

# %%


# %%


# %%
print("done!")
