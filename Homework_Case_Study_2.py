import matplotlib.pyplot as plt
import os
import pandas as pd
import numpy as np
from collections import Counter


def count_words_fast(text):
    text = text.lower()
    skips = [".", ",", ";", ":", "'", '"', "\n", "!", "?", "(", ")"]
    for ch in skips:
        text = text.replace(ch, "")
    word_counts = Counter(text.split(" "))   # Counter() returns a Python dict like object
    return word_counts


def word_stats(word_counts):
    num_unique = len(word_counts)
    counts = word_counts.values()
    return (num_unique, counts)


# read in the dataframe & set the first column in the csv file as the index
hamlets = pd.read_csv("C:\\Users\\Anoba\\Downloads\\hamlets.csv", index_col=1)

language,text = hamlets.iloc[0]   # selecting the row of English
# language is the first column of pandas df
# text is the second column of pandas df

counted_text = count_words_fast(text)
data = pd.DataFrame(columns=["Word","Count"])

len(counted_text.keys())
# 5113

n = 0
for word in counted_text.keys():
    data.loc[n] = word, counted_text[word]
    n += 1

data[data["Word"] == "hamlet"]   # 3  hamlet    97

length = []
for word in data["Word"]:
    length.append(len(word))
len(length)  # 5113

data["Length"] = length  # voila


freq = []
for c in data["Count"]:
    if c > 10:
        freq.append("Frequent")
    elif c > 1 and c <= 10:
        freq.append("Infrequent")
    else:
        freq.append("Unique")

data["Frequency"] = freq

unique_words = data.query("Frequency == 'Unique'")  # note the way arguments are passed
unique_words.shape  # (3348, 4)

# or

data.groupby('Frequency').count()  # voila

sub_data = pd.DataFrame({
    "language": language,
    "frequency": ["frequent", "infrequent", "unique"],
    "mean_word_length": data.groupby(by="frequency")["length"].mean(),
    "num_words": data.groupby(by="frequency").size()
})

# sub_data
# language	frequency	mean_word_length	num_words
# frequency
# frequent	English	frequent	4.371517	323
# infrequent	English	infrequent	5.825243	1442
# unique		English	unique	7.005675	3348

data.to_csv("Hamlet_python.csv")


def summarize_text(language, text):
    counted_text = count_words_fast(text)

    data = pd.DataFrame({
        "word": list(counted_text.keys()),
        "count": list(counted_text.values())
    })

    data.loc[data["count"] > 10,  "frequency"] = "frequent"
    data.loc[data["count"] <= 10, "frequency"] = "infrequent"
    data.loc[data["count"] == 1,  "frequency"] = "unique"

    data["length"] = data["word"].apply(len)

    sub_data = pd.DataFrame({
        "language": language,
        "frequency": ["frequent", "infrequent", "unique"],
        "mean_word_length": data.groupby(by="frequency")["length"].mean(),
        "num_words": data.groupby(by="frequency").size()
    })

    return(sub_data)


grouped_data = pd.DataFrame()
for row in range(0,3):
    language,text = hamlets.iloc[row]
    sub_data = summarize_text(language,text)
    grouped_data.append(sub_data)

# code provided in answer
grouped_data = pd.DataFrame(columns=["language", "frequency", "mean_word_length", "num_words"])
for i in range(hamlets.shape[0]):
    language, text = hamlets.iloc[i]
    sub_data = summarize_text(language, text)
    grouped_data = grouped_data.append(sub_data)
    
    
# visualization
colors = {"Portuguese": "green", "English": "blue", "German": "red"}
markers = {"frequent": "o", "infrequent": "s", "unique": "^"}
for i in range(grouped_data.shape[0]):
    row = grouped_data.iloc[i]
    plt.plot(row.mean_word_length, row.num_words, marker=markers[row.frequency], color=colors[row.language], markersize=10)
color_legend = []
marker_legend = []
for color in colors:
    color_legend.append(plt.plot([], [], color=colors[color], marker="o", label=color, markersize=10, linestyle="None"))
for marker in markers:
    marker_legend.append(plt.plot([], [], color="k", marker=markers[marker], label=marker, markersize=10, linestyle="None"))
plt.legend(numpoints=1, loc="upper left")
plt.xlabel("Mean Word Length")
plt.ylabel("Number of Words")
plt.show()


plt.savefig("translations.jpeg", dpi=2000)

os.listdir()
type(os.listdir())

title_num = 0
title_num ++
title_num += 1
title_num =+ 1
title_num + 1
