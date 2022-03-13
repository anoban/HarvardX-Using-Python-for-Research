# Matplotlib is a python plotting libarary
# Often the full Matplotlib isn't needed and the Pyplot happens to be enough in most situations
# Pyplot is a collection of functions that make Matplotlib work like Matlab

import matplotlib.pyplot as plt

# plt.plot() can be used to plot lines and markers
# simplest version of plot() takes one argument, which specifies the y axis values
# when values for only one axis is provides Python takes it for y values and takes the indices of 
# those values for x values
# since Python's indexing starts at 0, first element of the list/array appears at x = 0
# second element at x = 1, so on and so forth

import numpy as np

y = np.linspace(10, 150, 15)
plt.plot(y)  # note the x values

y = np.linspace(1, 20, 20)**2
plt.plot(y)

# to suppress plots getting printed a semicolon ";" at the end of .plot() call will do the job

plt.plot([1,4,9,16,25,36,49,64,81,100]);
# the plot still appears but the matplotlib object ([<matplotlib.lines.Line2D at xxxxxxxxxx>]) doesn't gets printed

# when working with IPython shell, plots are naturally shown once created
# an output in the form of [<matplotlib.lines.Line2D at xxxxxxxxx>] indicates that the plot has been created
# but in default Python's shell plt.show() needs to be issued to view the plot!

# plot() can take one/more arguments;
# where first argument represents the x coordinates
# second argument represents y coordinates

x = np.linspace(0,100, 100)
y = np.log2(x**3)
plt.plot(x,y)

z = x**2
plt.plot(x,z)  # a parabola results

# a third argument to the plot() call, which is a format string that specifies colour, marker &
# line type

# a keyword argument is an argument that is supplied to a function by explicitly naming each 
# parameter & specifying its values

# two important keyword arguments are linewidth & marker size

x = np.linspace(1,20,40)
y = np.sin(x)  # NumPy has its own sin() similar to math's sin()
z = np.cos(x)
plt.plot(x,y,"bo-")  
# b is used for blue colour 
# 0 is used for circles to be used as data points
# - is used for a solid line

plt.plot(x,y,"bo-", linewidth = 2, markersize = 4)
# marker size is the size of data points
# linewidth is the breadth of line connecting the data points

plt.plot(x,y,"go-", linewidth = 2.5, markersize = 10)  # green colour markers & line
plt.plot(x,y,"yo-", linewidth = 2, markersize = 4)   # yellow
plt.plot(x,y,"r*-", linewidth = 2, markersize = 4)   # stars as markers, red colour markers & lines

plt.plot(x,z,"bs-", linewidth = 2, markersize = 4)
# s is used to specify square markers

plt.plot([0,1,2],[0,1,4],"rd-")
# x = [0,1,2]
# y = [0,1,4]
# d represents diamond shaped markers

# some additional elements can easily be added to plots
# add a legend using legend() function
# ajust axes using axis() function
# set axes labels using xlabel() & ylabel()
# save the figure using savefig()

X = np.linspace(0,360,50)
Y = np.sin(X)
Z = np.cos(X)
plt.plot(X,Y,"ro-", linewidth = 2, markersize = 1.5)
plt.plot(X,Z,"go-", linewidth = 2, markersize = 1.5)
plt.xlabel("$Angle$")  
plt.ylabel("$Sin and Cos$")
plt.show()
# two plots together
# the $s wrapping the labels give an italic formatting to the labels

X = np.linspace(0,360,50)
Y = np.sin(X)
Z = np.cos(X)
plt.plot(X,Y,"ro-", linewidth = 2, markersize = 1.5)
plt.plot(X,Z,"go-", linewidth = 2, markersize = 1.5)
plt.xlabel("$Angle$")  
plt.ylabel("$Sin and Cos$")
plt.axis([-50,400,0,1])  
# arguments for the axis() call should be passed inside a list
# in the following order xmin, xmax, ymin, ymax
# in this case the axis() call slices the upper half of the full plot and extends the x axis
# on both sides beyond the min & max x values

# adding legends
X = np.linspace(0,360,50)
Y = np.sin(X)
Z = np.cos(X)
plt.plot(X,Y,"ro-", linewidth = 2, markersize = 1.5, label = "Sin")
plt.plot(X,Z,"go-", linewidth = 2, markersize = 1.5, label = "Cos")
plt.xlabel("$Angle$")  
plt.ylabel("$Sin and Cos$")
plt.axis([-50,400,-1,1]) 
plt.legend(loc = "upper left") 
# loc to specify the location of the legend
# "upper left" for upper left corner

X = np.linspace(0,360,50)
Y = np.sin(X)
Z = np.cos(X)
plt.plot(X,Y,"ro-", linewidth = 2, markersize = 1.5, label = "Sin")
plt.plot(X,Z,"go-", linewidth = 2, markersize = 1.5, label = "Cos")
plt.xlabel("$Angle$")  
plt.ylabel("$Sin and Cos$")
plt.axis([-50,400,0,1]) 
plt.legend(loc = "upper right")  # legend appears in the upper right corner

# to save the plot
X = np.linspace(0,360,50)
Y = np.sin(X)
Z = np.cos(X)
plt.plot(X,Y,"ro-", linewidth = 2, markersize = 1.5, label = "Sin")
plt.plot(X,Z,"go-", linewidth = 2, markersize = 1.5, label = "Cos")
plt.xlabel("$Angle$")  
plt.ylabel("$Sin and Cos$")
plt.axis([-50,400,0,1]) 
plt.legend(loc = "upper right")
plt.savefig("Sin_Cos_plot.jpeg")
# the file type is specified by the extension
# can be .pdf/.png/.svg etc

X = np.linspace(0,360,50)
Y = np.sin(X)
Z = np.cos(X)
plt.plot(X,Y,"ro-", linewidth = 2, markersize = 1.5, label = "Sin")
plt.plot(X,Z,"go-", linewidth = 2, markersize = 1.5, label = "Cos")
plt.xlabel("$Angle$")  
plt.ylabel("$Sin and Cos$")
plt.axis([-50,400,-1,1]) 
plt.legend(loc = "upper right")
plt.savefig("Sin_Cos_plot.jpeg", dpi = 6000)
# pixel quality of the output image can be increased by the dpi argument
# do not go over 1000 dpi, as session will crash due to memory overload - for Spyder
# 1000 dpi gives excellent pixel quality
# VSCode can handle dpi upto 6000 without any issues.

# to manually release memory in case of a likely crash
# import gc
# gc.collect()
# plt.close()
# plt.close("all") to remove all matplotlib objects

plt.close("all")
import gc
gc.collect()

# plotting using logarithmic axes
# either one or both axes can be logarithmic
# log values of x,y coordinates are considered when creating the plot
# by default base 10 log is used
# semilogx() plots the x axis on log scale and the y axis on the original scale
# semilogy() plots the y axis on log scale and the x axis on original scale
# loglog() plots both axes on log scale

# in an equation y = x^a 
# if a = 1, represents a line that goes through origin
x = np.linspace(0,1,100)
y = x
plt.plot(x,y,"bo-")
plt.show()
plt.close("all")

# a = .5 gives the square root
x = np.linspace(0,1,100)
y = x**.5
plt.plot(x,y,"bo-")
plt.show()

# a = 2 gives a parabola
x = np.linspace(0,1,100)
y = x**2
plt.plot(x,y,"gs-")
plt.show()

# in loglog() plot matplotlib takes the log of x and the log of y for plotting
# log(x**a) = a log(x)
# so in a loglog plot the equation used for plotting is log(y) = a log(x)
# say log(y) = y' and log(x) = x'
# now the equation is y' = a * x'  a much simpler code
# in loglog plots functions of y = x^a show up as straight lines
# with their gradient or slope giving the a value

x = np.array(range(1,21))
y = x**2
z = x**2.5
plt.plot(x, y, "gs-", markersize = 5, linewidth = 2, label = "x^2")
plt.plot(x, z, "bd-", markersize = 5, linewidth = 2, label = "x^2.5")
plt.xlabel("$X$")
plt.ylabel("$X^a$")
plt.legend(loc = "lower left")
plt.show()

# the avove plot as loglog plot

x = np.array(range(1, 21))
y = x**2
z = x**2.5
plt.loglog(x, y, color='red', marker='o', linestyle='-', markersize=3.5, linewidth=1.5, label="$x^2$")
plt.loglog(x, z, color='blue', marker='o', linestyle='-', markersize=3.5, linewidth=1.5, label="$x^2.5$")
plt.xlabel("$log(X)$")
plt.ylabel("$log(X^a)$")
plt.legend(loc="lower left")
plt.show()

# to make the points on x axis evenly spaced x coordinates can be supplied as log values
# logbase10(1) = 0, logbase10(20) = 1.30102999

x = np.logspace(0, 1.30102999, 20)
y = x**2
z = x**2.5
plt.loglog(x, y, color='red', marker='o', linestyle='-', markersize=3.5, linewidth=1.5, label="$x^2$")
plt.loglog(x, z, color='blue', marker='o', linestyle='-', markersize=3.5, linewidth=1.5, label="$x^2.5$")
plt.xlabel("$log(X)$")
plt.ylabel("$log(X^a)$")
plt.legend(loc="lower left")
plt.show()

# generating histograms

# np.random.normal() generates random numbers from a normal distribution
import matplotlib.pyplot as plt
import numpy as np

x = np.random.normal(size=1000)  # sample size is specified using the size argument
plt.hist(x)
plt.show()

# by default hist() uses 10 as bin width 

# density argument makes the histograms based on the proportion of values that fall under bins instead of 
# the count of values that fall under specified bins
x = np.random.normal(size=1000)
plt.hist(x, density=True, stacked=True) 
plt.show()

# bins agrument can be passed in to override defaults
x = np.random.normal(size=1000)
plt.hist(x, density=True, stacked=True, color="yellow", bins=np.linspace(-5,5,21))
plt.show()
# appearance of histogram changes with changes in binwidth

x = np.random.normal(size=1000)
plt.hist(x, density=True, stacked=True, color="teal", bins=np.linspace(-5, 5, 21))
plt.show()

# here we generate 21 points to create 20 bins
len(np.linspace(-5,5,21)) # 21
# these points specify the start and end of each bins
# so to have n bins n+1 points needs to be passed to the bins argument

# gamma distribution
# is a continuous probability function that starts at 0 & goes all the way up to positive infinity
# subplot() enables to have several subplots within each figure
# subplot() takes three arguments
# first 2 specify the number of rows and columns in the subplot
# and the third gives the plot number
# plot numbers always start at 1
# in a 2x3 subplot (there are 6 subplots in 2 rows and 3 columns)
# similar to R's grid arrange
# plot numbers increase row wise, here plot numbers are just labels for plots 

c = np.random.gamma(2,3,size=100000)  # drawing 100000 random samples from the specified gamma distribution

plt.figure()  # creates a figure to insert the histograms
plt.subplot(221)  #22 means 2x2 grid and the third digit 1 assigns this histogram as the 1st subplot
plt.hist(c, color = "red", bins=50)
plt.subplot(222)
plt.hist(c, color="red", bins=50, density = True, stacked = True) # to normalize the histogram
plt.subplot(223)
plt.hist(c, color="blue", bins=50, density=True, cumulative=True, stacked=True)  # cumulative proportions of bins
plt.subplot(224)
plt.hist(c, color="blue", bins=50, density=True, cumulative=True, stacked=True, histtype = "step") # step type stands for showing only the upper lines of bars without colour filling the bars
plt.savefig("subplot.jpeg", dpi=2000)

# to see the directory the figure was saved in
os.getcwd()

a = np.random.normal(size = 1000)
b = a**2
c = np.sqrt(b)
d = np.log10(b)
e = np.sin(a)
f = np.cos(a)

plt.figure()
plt.subplot(321)
plt.plot(a, color="red", linestyle="-", marker="o", markersize=3, linewidth=1.5, label="$RandomNormal$")
plt.subplot(322)
plt.plot(b, color="blue", linestyle="-", marker="o",markersize=3, linewidth=1.5, label="$sqr.RandomNormal$")
plt.subplot(323)
plt.plot(c, color="green", linestyle="-", marker="o", markersize=3, linewidth=1.5, label="$sqrt.RandomNormal$")
plt.subplot(324)
plt.plot(d, color="purple", linestyle="-", marker="o", markersize=3, linewidth=1.5, label="$log10.RandomNormal$")
plt.subplot(325)
plt.plot(e, color="violet", linestyle="-", marker="o", markersize=3, linewidth=1.5, label="$sin.RandomNormal$")
plt.subplot(326)
plt.plot(f, color="crimson", linestyle="-", marker="o", markersize=3, linewidth=1.5, label="$cos.RandomNormal$")
plt.legend(loc="best")
plt.savefig("six_subplots.jpeg", dpi=2500)

# separating the arguments passed in .subplot() works too

plt.figure() 
plt.subplot(2,2,1)
plt.hist(c, color="red", bins=50)
plt.subplot(2,2,2)
plt.hist(c, color="red", bins=50, density=True,stacked=True) 
plt.subplot(2,2,3)
plt.hist(c, color="blue", bins=50, density=True, cumulative=True, stacked=True)  
plt.subplot(2,2,4)
plt.hist(c, color="blue", bins=50, density=True, cumulative=True, stacked=True, histtype="step")
plt.show()
