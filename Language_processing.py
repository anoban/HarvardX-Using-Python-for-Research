# counting words
google = "Google LLC is an American multinational technology company that specializes in Internet-related services and products, which include online advertising technologies, a search engine, cloud computing, software, and hardware"

def count_words(text):
    """
    Count the number of time each unique word occurs in a given string
    """
    word_count = dict()
    for word in text.split(" "):
        if word in word_count:  # check if the word is already in the dictionary
            word_count[word] += 1  # adding 1 to occurances
        else:
            word_count[word] = 1  # new additions
    return word_count
   
count_words(google)  # works fine
# but includes punctuations
# this may lead to and | and, | ,and being counted separately
# and this function differentiates capitals i.e He and he are counted separately

# for this all characters in the input can be converted to lower case
# all possible punctuations can be contained in an iterable, which can be looped over to remove the punctuations

# redefining


def count_words(text):
    word_count = dict()
    skips = [",",".",";",":","'",'"']
    for punctuation in skips:
        text = text.replace(punctuation,"")
    for word in text.lower().split(" "):
        if word in word_count:  # check if the word is already in the dictionary
            word_count[word] += 1  # adding 1 to occurances
        else:
            word_count[word] = 1  # new additions
    return word_count

count_words(google)  # much better

# these tasks are so common that Python has a dedicated module for this
from collections import Counter

def word_counter(text):
    word_count = dict()
    text = text.lower()
    skips = [",",".",";",":","'",'"']
    for punctuation in skips:
        text = text.replace(punctuation,"")
    word_counts = Counter(text.split(" "))   # using the counter module
    return word_counts
    
word_counter(google)  # voila

output1 = count_words(google)
output2 = word_counter(google)

output1 == output2  # True
type(output1)  # <class 'dict'>
type(output2)  # <class 'collections.Counter'>


len(count_words("This comprehension check is to check for comprehension."))  # 6

# character encoding means how computer encodes certain characters
# UTF-8 encoding is the most widely used encoding used in web

def read_book(path):
    """
    This will read a book and return it as a string
    """
    with open(path, "r", encoding="utf8") as red_book:
        text = red_book.read()
        text = text.replace("\n","").replace("\r","")
    return text


text = read_book("C:\\Users\\Anoba\\Documents\\Python\\Using_Python_for_Research\\Gutenberg\\Books_EngFr\\Books_EngFr\\English\\shakespeare\\Romeo and Juliet.txt")
len(text)  # 169275

text.find("What's in a name?")   # 42757
# find method will return an index if it finds a substring

sample = text[42757:42757+1000]
sample   # "What's in a name? That which we call a rose......"

def word_stats(word_counts):
    """
    Returns the number of unique words and word frequencies
    """
    num_unique = len(word_counts)
    counts = word_counts.values()
    return(num_unique,counts)
   
word_counts = word_counter(text)   
word_stats(word_counts)    # 5118 unique words

(unique_words,frequencies) = word_stats(word_counts)
sum(frequencies)   # 40776

german_text = read_book("C:\\Users\\Anoba\\Documents\\Python\\Using_Python_for_Research\\Gutenberg\\Books_GerPort\\Books_GerPort\\German\\shakespeare\\Romeo und Julia.txt")
german_word_counts = word_counter(german_text)
(unique_german_words, german_frequencies) = word_stats(german_word_counts)
unique_german_words  # 7527 unique words
sum(german_frequencies)  # 20311

# reading multiple files
import os
book_dir = "C:\\Users\\Anoba\\Documents\\Python\\Using_Python_for_Research\\Gutenberg\\"
os.listdir(book_dir)  # tells the present directory contains 4 items 
# ['English', 'French', 'German', 'Portuguese'] this contains the paths for all 4 subdirectories
# this isn't a list; a special list like object

for language in os.listdir(book_dir):  # loops over language level
    for author in os.listdir(book_dir + language):  # loops over author level inside each languages
        for title in os.listdir(book_dir + language + "\\" + author):  # loops over title level inside all authors
            path = book_dir + language + "\\" + author + "\\" + title
            print(path)
            book_text = read_book(path)
            (unique_words, frequencies) = word_stats(word_counter(book_text)) 
            
# see the results

# pandas is a library that provides additional data structures and data analysis functinalities
# pandas is particularly useful in manipulating numerical tabls and timeseries data
# pandas gets its name from PANel DAta used to refer to multi dimensional data structures

import pandas as pd
# creating a table using pandas' dataframe function
table = pd.DataFrame(columns=["name","age"]) 
table.loc[1] = "James",43          
table.loc[2] = "Julia", 31
table.loc[3] = "Natalie", 27
table.loc[4] = "Sam", 23
table.loc[5] = "Rosalia", 19
table.loc[6] = "Zendaya", 25
table.loc[7] = "Emma", 22

table  # to inspect the table
table.columns  # to inspect the column names

stats = pd.DataFrame(columns=["language","author","title","length","unique"])
title_no = 1
for language in os.listdir(book_dir):  
    for author in os.listdir(book_dir + language):
        for title in os.listdir(book_dir + language + "\\" + author):
            path = book_dir + language + "\\" + author + "\\" + title
            print(path)
            book_text = read_book(path)
            (unique_words, frequencies) = word_stats(word_counter(book_text))
            stats.loc[title_no] = language,author,title,sum(frequencies), unique_words
            title_no += 1

stats.shape  # (102, 5)
stats.head()  # top 5 rows
stats.tail()  # bottom 5 rows

# its better to capitalize the firts letter of authors
# remove the .txt extensions from the title column
import time
start = time.time()
stats = pd.DataFrame(columns=["language", "author", "title", "length", "unique"])
title_no = 1
for language in os.listdir(book_dir):
    for author in os.listdir(book_dir + language):
        for title in os.listdir(book_dir + language + "\\" + author):
            path = book_dir + language + "\\" + author + "\\" + title
            print(path)
            book_text = read_book(path)
            (unique_words, frequencies) = word_stats(word_counter(book_text))
            stats.loc[title_no] = language, author.capitalize(), title.replace(".txt",""), sum(frequencies), unique_words
            title_no += 1
end = time.time()
print("Parsing time = ", end-start)   # Parsing time =  2.950592279434204
# perfecto

stats.length  # gives the length column, similar to R's $
stats.unique

import matplotlib.pyplot as plt
plt.plot(stats.length, stats.unique, color = "red", marker="o", markersize=10, linestyle="None")
plt.xlabel("$Length\hspace{1}of\hspace{1}unique\hspace{1}words$")
plt.ylabel("$Unique\hspace{1}words$")
plt.show()
# linestyle="None" for a scatter plot

plt.loglog(stats.length, stats.unique, color="blue", marker="o", markersize=10, linestyle="None")
plt.xlabel("$Length\hspace{1}of\hspace{1}unique\hspace{1}words\hspace{1}in\hspace{1}log\hspace{1}scales$")
plt.ylabel("$Unique\hspace{1}words\hspace{1}in\hspace{1}log\hspace{1}scales$")
plt.title("Distribution of unique words and corresponding word lengths")
plt.show()
# almost a straight line

e = stats[stats.language == 'English']
f = stats[stats.language == 'French']
g = stats[stats.language == 'German']
p = stats[stats.language == 'Portuguese']

plt.figure(figsize=(64,64))
plt.plot(e.length,e.unique, marker="o", markersize=8, color="crimson", label="English", linestyle="None")
plt.plot(f.length, f.unique, marker="o", markersize=8, color="blue", label="French", linestyle="None")
plt.plot(g.length, g.unique, marker="o", markersize=8, color="teal", label="German", linestyle="None")
plt.plot(p.length, p.unique, marker="o", markersize=8, color="orange", label="Portuguese", linestyle="None")
plt.legend(loc="upper right")
plt.xlabel("$Number\hspace{1}of\hspace{1}words$")
plt.ylabel("$Number\hspace{1}of\hspace{1}unique\hspace{1}words$")
plt.title("$Distribution\hspace{1}of\hspace{1}total\hspace{1}word\hspace{1}count\hspace{1}and\hspace{1}number\hspace{1}of\hspace{1}unique\hspace{1}words\hspace{1}in\hspace{1}literatures\hspace{1}from\hspace{1}different\hspace{1}languages$")
plt.show(block=None)



















