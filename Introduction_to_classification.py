# statistical learning
# supervised learning - estimate/predict an output based on one/more inputs
# inputs are called various names; predictors, independent variables, features, variables
# outputs are called; response variables, dependent variables

# if the response is quantitative, the problem is called regression problems
# if the response is qualitative (categorical), the problem is called classification problems


# k nearest neighbour classifier (kNN)
# given a positive integer k, and a new data point kNN classifier first identifies those k points in the data,
# that are nearest to the point, and classifies the new data point as belonging to the most common class
# among those k neighbours

# eg. we have a dataset with data points of peoples height and weight
# 4 classes are defined; short and thin, short and obese, tall and thin & thin and obese
# if a new data point (height & weight of a new person) is supplied the kNN classified decides to which
# class the new person belongs and assigns him/her to that class accodingly

# kNN first identifies the closest neighbours in the existing dataset for the new input 
# then assigns the new datapoint to the category to which majority of the neighbouring points belong

# finding the distance between two points
# distance here means the Euclidean distance

# data point a with coordinates (x1,y1)
# data point b with coordinates (x2,y2)
# Euclidean distance between a & b
# distance on x axis |x1-x2| (to avoid problems whenever x1 < x2,  absolute value is taken) 
# distance on y axis |y1-y2|
# Pythogaras's theorem; euclidean distance d = ( (|x1-x2|)^2 + (|y1-y2|)^2 )^0.5
# since we are squaring the differences here it doesn't matter whether x1 < x2 or vice versa
# the result is the same

import numpy as np
# define 2 data points
p1 = np.array([1,1])
p2 = np.array([4,4])
p2-p1  # array([3, 3]) gives the distances axes wise
np.power(p1-p2, 2) # ^2
# array([9, 9], dtype=int32) - square of axiswise distances

np.sum(np.power(p1-p2, 2))  # 18 - sum of square of axiswise distances
np.sqrt(np.sum(np.power(p1-p2, 2)))
# 4.242640687119285 - square root of sum of square of axiswise distances

def distance(p1,p2):
    """Finds the Euclidean distance between the two supplied two dimensional coordinates"""
    import numpy as np
    if len(p1) and len(p2) == 2:
        a = np.array(p1)
        b = np.array(p2)
        return np.sqrt(np.sum(np.power(a-b, 2)))
    else:
        return "This function works only with two dimensional coordinates"
  
      
distance(p1, p2)  # 4.242640687119285 - works
distance((1, 3, 2), (4, 5, 3))
# 'This function works only with two dimensional coordinates'


sample_dict = {85: "Julia", 93: "Marie", 78: "Natalie"}
max(sample_dict)     # 93 this gives the maximum key value
# equivalen to
max(sample_dict.keys())  # 93
max(sample_dict.values())  # Natalie - the longest name!!!

for marks,student in sample_dict.items():
    print(marks,student)
# 85 Julia
# 93 Marie
# 78 Natalie

seq = [3, 3, 3, 3, 4, 5, 6, 7, 7, 7, 7, 8, 9, 1]
max(seq)  # 9

# majority vote
# we need to find the number of times each element occurs and need to find the most frequent element

def majority_vote(sequence_object):
    import random
    vote_counts = dict()
    highest_f = []  # in case multiple elements have the highest frequency
    for vote in sequence_object:
        if vote in vote_counts:
            vote_counts[vote] += 1
        else:
            vote_counts[vote] = 1
    max_vote_count = max(vote_counts.values())
    for vote,count in vote_counts.items():
        if count == max_vote_count:  # if the selected frequency is that of the vote with maximum frequency
            highest_f.append(vote)
    return random.choice(highest_f) # when there are many elements with highest frequencies, randomly selecting any one will be good enough
    
    
majority_vote(seq) # 7,3 can alternate!

# most commonly ocurring element in a distribution is called a mode in statistics
# scipy has a built-in function


# redefine the function using scipy
def majority_vote_short(iterable):
    import scipy.stats as ss
    mode,count = ss.mstats.mode(iterable) # scipy.stats.mstats.mode() returns mode & its frequency
    return mode

majority_vote_short(seq) # however scipy mode returns the lowest mode when there is a tie


# finding kNN

# pseudocode
# loop over all data points in the available dataset
# compute distances between the unknown point & all existing points
# sort the distances
# find the nearest neighbours i.e points with shortest distances

points = np.array([[1,1],[1,2],[1,3],[2,1],[2,2],[2,3],[3,1],[3,2],[3,3]])
points.shape # 9x2

p = np.array([2.5,2]) # unknown point

import matplotlib.pyplot as plt
plt.plot(points[:,0],points[:,1], marker="o", markersize=10, color="red", linestyle="None")
plt.plot(p[0], p[1], marker="o", markersize=10, color="green", linestyle="None")
plt.show()

distances = np.zeros(points.shape[0])  # points.shape[0] = 9
for i in range(0,len(distances)):
    distances[i] = distance(p, points[i])
    
distances  # array([1.80277564, 1.5, 1.80277564, 1.11803399, 0.5, 1.11803399, 1.11803399, 0.5, 1.11803399])

np.argsort(distances)  # array([4, 7, 3, 5, 6, 8, 1, 0, 2], dtype=int64)
# numpy.argsort() sorts the elements of the array and returns the indices of the sorted elements 
# in the sorted order!!

ascen_ind = np.argsort(distances)
distances[ascen_ind]
# array([0.5, 0.5 , 1.11803399, 1.11803399, 1.11803399,1.11803399, 1.5, 1.80277564, 1.80277564])

# to pick the first 3 points closest to p
distances[ascen_ind[0:3]]
# array([0.5, 0.5 , 1.11803399])

# define this as a function

def nearest_neighbour(p,points,k=5):
    import numpy as np
    """Takes an unknown point - p, existing dataset - points and 
    the needed number of nearest neighbours - k to return k number of nearest
    neighbours to the unknown point p"""
    distances = np.zeros(points.shape[0])
    for i in range(0, len(distances)):
        distances[i] = distance(p, points[i])
    return np.argsort(distances)[0:k] # returns the indices of k nearest neighbours


ind = nearest_neighbour((4,2.1),points,2); print(points[ind])  # array([[3, 2], [3, 3]])
ind = nearest_neighbour((4, 2.1), points, 4); print(points[ind])  # array([[3, 2],[3, 3],[3, 1],[2, 2]])
 
# when a parameter of a function is intrinsically specified inside a function call, Python will default 
# to that value when that parameter isn't passed to the function
# in the following case k=5!!

def kNN_predict(p,points,outcomes,k=5):  # here points, outcomes are the training datasets
    ind = nearest_neighbour(p,points,k)
    return majority_vote(outcomes[ind])
   
# points had 9 elements, thus outcomes must have 9 elements    
# outcomes is a training dataset that contains the classes of input elements
# so Python finds the nearest neighbour, finds its index, and then finds its corresponding class from outcomes
# then classifies the new datapoint under the class of nearest neighbour!!
outcomes = np.array([0,0,0,0,1,1,1,1,1])  
len(outcomes) == len(points)  # True

kNN_predict(np.array([2.5,2.7]),points,outcomes,2)  # 1
# here's what happens here
# a new datapoint with coordinates x=2.5, y=2.7 are passed to the function
# Python is asked to classify this point
# we have 2 different classes here; class 1 & class 0

kNN_predict(np.array([1.0,2.7]),points,outcomes,2)  # 0


# generating synthetic data
# generating 2 end datapoints, first one from class 0 & the second one from class 1 (outputs)
# these data are known as synthetic data, since they're generated with the help of computer
# datapoints from two bivariate normal distributions
# bivariate means two variables e.g. x,y

import scipy.stats as ss
ss.norm(0,1).rvs((5,2))  # first argument is mean, second is sd, and the dimension of output array is
# passed in the rvs()
# rvs() stands for random variates
ss.norm(1, 1).rvs((5, 2))

# concatanate the 2 above arrays to get a single array with 2 columns and 10 rows
np.concatenate((ss.norm(0,1).rvs((5,2)),ss.norm(1, 1).rvs((5, 2))), axis=0)  # voila

# to generate outcomes; first 10 belong to class 0
# second 10 outcomes belong to class 1

np.repeat(0,10)  # repeats the given value given times
np.repeat(1,10)
np.concatenate((np.repeat(0, 10), np.repeat(1, 10)))

# to turn this into a function

def gen_syn_data(n=50):
    """Creates 2 sets of points from bivariate normal distributions
    """
    import numpy as np
    import scipy.stats as ss
    class_0 = ss.norm(0, 1).rvs((n, 2))
    class_1 = ss.norm(1, 1).rvs((n, 2))
    points = np.concatenate((class_0,class_1), axis=0)
    outcomes = np.concatenate((np.repeat(0, n), np.repeat(1, n)))
    return (points,outcomes)

gen_syn_data(15)
points,outcomes = gen_syn_data(50)

import matplotlib.pyplot as plt
plt.plot(points[:n,0], points[:,1], color="blue", marker="o", markersize=5,linestyle="-", label="$Class\hspace{1}0$")  # elements up to n
plt.plot(points[n:,0], points[n:,1], color="crimson", marker="o", markersize=5,linestyle="-", label="$Class\hspace{1}1$")  # elements from n
plt.legend(loc="lower right")
plt.show()

n = 50
plt.plot(points[:n, 0], points[:n, 1], color="blue", marker="o", markersize=10, linestyle="None", label="$Class\hspace{1}0$")  # elements up to n
plt.plot(points[n:, 0], points[n:, 1], color="crimson", marker="o", markersize=10, linestyle="None", label="$Class\hspace{1}1$")  # elements from n
plt.legend(loc="lower right")
plt.show()

# making a prediction grid
# once wev'e observed our data, we can examine some part of the predictor space, and compute
# the class prediction for each point in th grid using the knn classifier
# instead of finding how the knn classifier might classify the given point, we can see how it will 
# classify all points that belong to a rectangular region of the predictor space

# h - step size
# limits must be passed as a tuple

def make_prediction_grid(predictors, outcomes, limits, h, k):
    """Classifies each point on the prediction grid."""
    (x_min, x_max, y_min, y_max) = limits  # tuple unpacking
    import numpy as np
    xs = np.arange(x_min,x_max,h)  # x axis of the grid
    ys = np.arange(y_min,y_max,h)  # y axis of the grid
    xx,yy = np.meshgrid(xs,ys) 
    # classifier predictions corresponding to every point in the meshgrid
    prediction_grid = np.zeros(xx.shape, dtype=int)  # or the shape of  yy
    for i,x in enumerate(xs):
        for j,y in enumerate(ys):
            p = np.array([x,y])
            prediction_grid[j,i] = kNN_predict(p, predictors, outcomes, k)
    return (xx, yy, prediction_grid)

# y corresponds to y values and j corresponds to indices of those y values
# x corresponds to x values and i corresponds to indices of those y values
# in prediction grid j,i are provided as row and column indices
# indices of ys are given for rows and indices of xs for columns

# meshgrid takes in 2/more coordinate vectors, with one vector having the x values of interest and the other vector
# having y values of interest
# it returns matrices, first containing x values for each grid point, second containing the y values for each 
# grid point


np.meshgrid(np.arange(0, 5, 1), np.arange(0, 5, 1))
# [array([[0, 1, 2, 3, 4],[0, 1, 2, 3, 4],[0, 1, 2, 3, 4],[0, 1, 2, 3, 4],[0, 1, 2, 3, 4]]), 
# array([[0, 0, 0, 0, 0],[1, 1, 1, 1, 1],[2, 2, 2, 2, 2],[3, 3, 3, 3, 3],[4, 4, 4, 4, 4]])]
# first array is the x coordinates of all points of a mesh grid with x values 0 to 4
# second array is the y coordinates of all points of a mesh grid with y values 0 to 4

seasons = ["Summer","Winter","Autumn","Spring"]
for ind,season in enumerate(seasons):
    print(ind,season)
    
# 0 Summer
# 1 Winter
# 2 Autumn
# 3 Spring

enumerate(seasons) # Python creates an enumerate object
# to view the enumerate object
list(enumerate(seasons))
# [(0, 'Summer'), (1, 'Winter'), (2, 'Autumn'), (3, 'Spring')]
# enumerate returns a sequence of tuples.


# plotting the prediction grid
# code for the function is provided as course material
def plot_prediction_grid(xx, yy, prediction_grid, filename):
    """ Plot KNN predictions for every point on the grid."""
    from matplotlib.colors import ListedColormap
    background_colormap = ListedColormap(["hotpink", "lightskyblue", "yellowgreen"])
    observation_colormap = ListedColormap(["red", "blue", "green"])
    plt.figure(figsize=(10, 10))
    plt.pcolormesh(xx, yy, prediction_grid,cmap=background_colormap, alpha=0.5)
    plt.scatter(predictors[:, 0], predictors[:, 1],c=outcomes, cmap=observation_colormap, s=50)
    plt.xlabel('Variable 1')
    plt.ylabel('Variable 2')
    plt.xticks(())
    plt.yticks(())
    plt.xlim(np.min(xx), np.max(xx))
    plt.ylim(np.min(yy), np.max(yy))
    plt.savefig(filename)

(predictors,outcomes) =  gen_syn_data(100)
k=5; filename="knn_k5.pdf"; limits=(-3,4,-3,4); h=0.1
(xx, yy, prediction_grid) = make_prediction_grid(predictors,outcomes,limits,h,k)
plot_prediction_grid(xx,yy,prediction_grid, filename)

(predictors,outcomes) =  gen_syn_data(100)
k=50; filename="knn_k50.pdf"; limits=(-3,4,-3,4); h=0.1
(xx, yy, prediction_grid) = make_prediction_grid(predictors,outcomes,limits,h,k)
plot_prediction_grid(xx,yy,prediction_grid, filename)

# as k increases the decision boundary becomes smoother
# for smaller k values the decision boundary is more rugged

# applying the knn method
# using scikit-learn knn classifier

from sklearn import datasets
iris = datasets.load_iris()  # a df embedded in the sklearn module
# object iris is not a dataframe
# to view the dataframe
iris["data"]
iris["data"].shape  # (150, 4) 150 rows and 4 columns

predictors = iris.data[:,0:2] # all rows & first and second columns
outcomes = iris.target
plt.plot(predictors[outcomes == 0][:, 0], predictors[outcomes == 0][:, 1], "ro")  # covers the datapoints from class 0
plt.plot(predictors[outcomes == 1][:, 0], predictors[outcomes == 1][:, 1], "go")  # covers the datapoints from class 1
plt.plot(predictors[outcomes == 2][:, 0], predictors[outcomes == 2][:, 1], "bo")  # covers the datapoints from class 2
plt.show()

# making a prediction grid plot
predictors = iris.data[:, 0:2] 
outcomes = iris.target
k = 5; filename = "sklearn_knn_k5.pdf"; limits = (4, 8, 1.5, 4.5); h = 0.1
(xx, yy, prediction_grid) = make_prediction_grid(predictors, outcomes, limits, h, k)
plot_prediction_grid(xx, yy, prediction_grid, filename)


# comparing the above knn classifier & sklearn knn classifier
from sklearn.neighbors import KNeighborsClassifier as knn
knn_5 = knn(n_neighbors=5)
knn_5.fit(predictors,outcomes)
sk_predictions = knn_5.predict(predictors)
# sk_predictions.shape  (150,)

my_predictions = np.array([kNN_predict(p,predictors,outcomes,5) for p in predictors])
# my_predictions.shape  (150,)

sk_predictions == my_predictions
sum(sk_predictions == my_predictions)  # 146
# 146 times out of 150 my_predictions agree with sk_predictions

np.mean(sk_predictions == my_predictions)  # 0.9733333333333334
np.mean(sk_predictions == my_predictions)*100  # 97.33333333333334 %

plt.plot(sk_predictions, "bo", label="$SK\hspace{1}Learn\hspace{1}predictions$")
plt.plot(my_predictions, "ro", label="$My\hspace{1}predictions$")
plt.xlabel("$Indices$")
plt.ylabel("$Predicted\hspace{1}values$")
plt.legend(loc="lower right")
plt.show()

np.mean(my_predictions == outcomes)  # 0.8466666666666667
np.mean(sk_predictions == outcomes)  # 0.8333333333333334

























