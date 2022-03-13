import numpy as np
import random
import matplotlib.pyplot as plt
from re import X
import scipy.stats as ss
import pandas as pd
import sklearn.preprocessing
import sklearn.decomposition


def majority_vote_fast(votes):
    mode, count = ss.mstats.mode(votes)
    return mode

def distance(p1, p2):
    return np.sqrt(np.sum(np.power(p2 - p1, 2)))

def find_nearest_neighbors(p, points, k=5):
    distances = np.zeros(points.shape[0])
    for i in range(len(distances)):
        distances[i] = distance(p, points[i])
    ind = np.argsort(distances)
    return ind[:k]

def knn_predict(p, points, outcomes, k=5):
    ind = find_nearest_neighbors(p, points, k)
    return majority_vote_fast(outcomes[ind])[0]


wines = pd.read_csv("C:\\Users\\Anoba\\Documents\\Python\\Using_Python_for_Research\\wines.csv", header=0)
wines.head()

len(wines.color)  # 6497

# to convert the color column to numbers; red = 1, other colors = 0
for i in range(6497):
    if wines.color[i] == "red":
        wines.color[i] = 1
    else:
        wines.color[i] = 0
    
wines = wines.rename(columns={"color":"is_red"}) # renamed the color column to is_red

sum(wines.is_red)  # 1599

# removing non numeric columns from the dataset
# pandas.dataframe.drop() allows to remove rows/columns

numeric_data = wines.drop(columns=["quality", "high_quality"])
scaled_data = pd.DataFrame(sklearn.preprocessing.scale(numeric_data), columns=numeric_data.columns)

pca = sklearn.decomposition.PCA()
principal_components = pca.fit_transform(scaled_data)  # an array results

principal_components.shape  # (6497, 13)



from matplotlib.colors import ListedColormap
from matplotlib.backends.backend_pdf import PdfPages
observation_colormap = ListedColormap(['red', 'blue'])
x = principal_components[:,0]
y = principal_components[:,1]

plt.title("Principal Components of Wine")
plt.scatter(x, y, alpha = 0.2, c = wines['high_quality'], cmap = observation_colormap, edgecolors = 'none')
plt.xlim(-8, 8); plt.ylim(-8, 8)
plt.xlabel("Principal Component 1")
plt.ylabel("Principal Component 2")
plt.show()

np.random.seed(1) # do not change this!
x = np.random.randint(0, 2, 1000)
y = np.random.randint(0 ,2, 1000)

def accuracy(predictions, outcomes):
    comparison = []
    if len(predictions) == len(outcomes):
         for i in range(len(predictions)):
             comparison.append(predictions[i]==outcomes[i])
    else:
        print("Both iterables should be of same length")
    return np.mean(comparison)*100

accuracy(x,y)  # 51.5

np.mean(x==y)  # 0.515

def accuracy(predictions, outcomes):
    return np.mean(predictions==outcomes)*100

accuracy(0, wines["high_quality"])  # 36.69385870401724

from sklearn.neighbors import KNeighborsClassifier
knn = KNeighborsClassifier(n_neighbors = 5)
knn.fit(scaled_data, wines['high_quality'])

library_predictions = knn.predict(scaled_data)
accuracy(wines['high_quality'], library_predictions)  # 84.20809604432816

random.seed(123)
n_rows = wines.shape[0]
selection = random.sample(range(n_rows), 10)