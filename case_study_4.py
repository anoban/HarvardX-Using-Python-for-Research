import numpy as np
import random
import matplotlib.pyplot as plt
import scipy.stats as ss
import pandas as pd
import sklearn.preprocessing
import sklearn.decomposition

# a dataset of scotch whiskies
# contains taste ratings of one readily available single malt scotch whisky from almost all whisky distilery 
# in scotland
# 86 malt whiskies, scored between 0 to 4 
# in 12 different taste categories
#  scores has been collected from 10 different tasters
# pandas has two principal data types;
# series and dataframes
# series is 1 D array like object
# dataframe is a 2 D array like object

x = pd.Series([5,4,3,9,45,12,7])
# each element in x has an index, since we didn't explicitly specify an idex Python's default index is 
# assigned to the series elements

students = pd.Series(["James","Julia","Janice","Natalie","Sarah","Cassandra"], index=["J","Ju","Ja","Na","Sa","Ca"])
students # a pandas series object with explicitly declared strings as indices

students["J"]  # 'James'
students["Na"]  # 'Natalie'

students[["J","Ja","Na"]]  # very much like numpy arrays
# J       James
# Ja     Janice
# Na    Natalie

# pandas series object can be created using dictionaries

my_dict = {"Jim":25, "Pam":41, "Sam":36, "Leslie":29}

my_series = pd.Series(my_dict)
my_series
# Jim       25
# Pam       41
# Sam       36
# Leslie    29
# indices of the series consists of the keys of the dict object (in alphabetically sorted order)
# and the elements of the series are the values of the dict object

# dataframes have both column and row indices

df_dict = {"Name":["Luie","Sarah","Nathan","Cassandra"], "Age":[19,25,18,24], "ZIP":["0125","4587","0102","7800"]}
# keys = Nmae,Age & ZIP
# values = lists

df_dict.keys()  # dict_keys(['Name', 'Age', 'ZIP'])
df_dict.values()
# dict_values([['Luie', 'Sarah', 'Nathan', 'Cassandra'], [19, 25, 18, 24], ['0125', '4587', '0102', '7800']])

for cols,records in df_dict.items():
    print(cols,records)
# Name ['Luie', 'Sarah', 'Nathan', 'Cassandra']
# Age[19, 25, 18, 24]
# ZIP['0125', '4587', '0102', '7800']

data_frame = pd.DataFrame(df_dict, columns=df_dict.keys())
data_frame
#          Name  Age   ZIP
# 0       Luie   19  0125
# 1      Sarah   25  4587
# 2     Nathan   18  0102
# 3  Cassandra   24  7800

# Python indexed the dataframe with its defaults since we didn't specify the row indices explicitly

data_frame["Name"] # returns the Names column
data_frame["Name"][3]  # 'Cassandra'
data_frame.Name # voila

# rearranging the indices of pandas objects
my_series # indices are names
my_series.index
# Index(['Jim', 'Pam', 'Sam', 'Leslie'], dtype='object')

sorted(my_series.index)  # ['Jim', 'Leslie', 'Pam', 'Sam']

# reindexing
my_series.reindex(index=sorted(my_series.index))
# Jim       25
# Leslie    29
# Pam       41
# Sam       36

# when adding/subtracting two/more pandas series objects, operations happens by index
# similar to numpy arrays
# if the indices do not match pandas introduces NAN (Not a number) objects

series1 = pd.Series(["L","M","N","O"], index=["l","m","n","o"])
series2 = pd.Series(["P", "Q", "R", "S", "T"], index=["p", "q", "r", "s", "t"])
len(series1) # 4
len(series2) # 5

series2 + series2
# p    PP
# q    QQ
# r    RR
# s    SS
# t    TT

series1 + series2
# l    NaN
# m    NaN
# n    NaN
# o    NaN
# p    NaN
# q    NaN
# r    NaN
# s    NaN
# t    NaN

series3 = pd.Series(["L", "M", "N", "Q"], index=["l", "m", "n", "q"])
series1 + series3
# l     LL
# m     MM
# n     NN
# o    NaN
# q    NaN
# NANs are introduced where the indices do not match

# these index wise arithmetic operations work the same way with dataframe objects as well

whiskies = pd.read_csv("C:\\Users\\Anoba\\Documents\\Python\\Using_Python_for_Research\\whiskies.txt")
whiskies

regions = pd.read_csv("C:\\Users\\Anoba\\Documents\\Python\\Using_Python_for_Research\\regions.txt")
regions

whiskies.shape  # (86, 17)
regions.shape  # (86, 1)

whiskies["Region"] = regions # adding the regions dataframe (column) to the whiskies dataframe
whiskies

whiskies.columns
# ['RowID','Distillery', 'Body', 'Sweetness', 'Smoky', 'Medicinal', 'Tobacco','Honey', 'Spicy', 'Winey', 'Nutty', 'Malty', 'Fruity', 'Floral','Postcode', ' Latitude', ' Longitude', 'Region']

# iloc method
whiskies.iloc[5:16] # rows from 5 to 15
whiskies.iloc[5:11,1:5] # rows from 5 to 10 and columns from 1 to 4

whiskies

# to create a flavours dataframe (dataframe slicing)

for ind,col in enumerate(whiskies.columns):
    print(ind,col)
      
flavours = whiskies.iloc[:,2:14] # all the rows, and columns specific to flavours
flavours

# exploring correlations
# by default Pearson correlation coefficient is computed
# this is a linear correlation coeficient that can range from -1 to +1
# +1 means the 2 variable tend to increase/decrease together - strong positive correlation
# -1 means that as one variable increases the other decreases - strong negative correlation
# 0 means that the  2 variables are independent; not correlated at all

flavour_correlations = pd.DataFrame.corr(flavours) 
# corr() function of pandas computes the linear correlation coefficient by default

plt.figure(figsize=(50,50))
plt.pcolor(flavour_correlations) # pseudo colour plot
plt.colorbar()
plt.show()
# shows the pearson correlation of different flavours

# to inspect the correlations between different whiskies based on their flavour profile
# how similar are different whiskies made by different distilleries in Scotland
# .transpose() turns rows to columns and columns to rows

flavours.transpose()
# flavours as rows
# ratings as columns

plt.figure(figsize=(50, 50))
plt.pcolor(pd.DataFrame.corr(flavours.transpose())) 
plt.colorbar()
plt.show()

# plt.axis("tight") removes unnecessary whitespaces inside the plotting pane

# clustering whiskies by flavour profile
# for example
# we have a set of words and a set of documents
# compute the frequency of these words in each of these documents
# clustering is used to find sets of words and sets of documents that often go together
# words belonging to certain fields of study can be commonly found in articles related to that field
# eg. "mitochondria" will more frequently occur in biology articles than in physics articles 
# so we cluster "mitochondria" with biology

# co-clustering means finding the clusters of words and clusters of documents simultaneously
# thus simultaneously clustering the rows and columns of dataframes/matrices

from sklearn.cluster import SpectralCoclustering
model = SpectralCoclustering(n_clusters=6, random_state=0) # creating a model
# n_clusters=6 means that we are asking for 6 clusters; from 0 to 5
# second step is to fit the model
model.fit(pd.DataFrame.corr(flavours.transpose())) 
model.rows_  # rows of the model
# each entry is a boolean
# each row in this array identifies a cluster ranging from 0 to 5
# each column identifies a row in the correlation matrix ranging from 0 to 85

# to see how many whiskies belong to each clusters
np.sum(model.rows_, axis=1) # columnwise sums
# array([20,  5, 19, 17,  6, 19])

# to inspect to which cluster the whiskies belong
model.row_labels_  # returns an array of index wise cluster labels

# comparing correlation matrices
# appending the cluster labels to whiskies dataframe

type(model.row_labels_)  # <class 'numpy.ndarray'>
# converting the above object to pandas series object will allow for explicit indexing
whiskies["Clusters"] = pd.Series(model.row_labels_, index= whiskies.index)
whiskies.rename(columns={"Clusters":"Group"}) # whoopsies

np.argsort(model.row_labels_)  # sorts the model.row_labels_ array and returns the indices of the elements
# in sorted order

sorted(model.row_labels_)  # ascending order

# reordering rows in whiskies dataframe according to clusters
whiskies = whiskies.iloc[np.argsort(model.row_labels_)]
whiskies = whiskies.reset_index(drop=True) # resetting the row indices

# computing the correlating matrix again, after cluserwise sorting of records
pd.DataFrame.corr(whiskies.iloc[:,2:14].transpose())
ordered_correlations = np.array(pd.DataFrame.corr(whiskies.iloc[:, 2:14].transpose()))

plt.figure()
plt.subplot(121)
plt.pcolor(pd.DataFrame.corr(flavours.transpose()))
plt.title("$Original$")
plt.subplot(122)
plt.pcolor(ordered_correlations)
plt.title("$Clustered$")
plt.show()
