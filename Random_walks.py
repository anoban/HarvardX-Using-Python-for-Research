# a walker starts his walk at time t=0
# walk starts from origin (0,0)
# his position at time t=0 is denoted as p(t=0) = x
# his position the next seceond (t=1) is his position at time t=0 + the dispalcement that
# happened at time t=1 say dx(t=1)
# p(t=1) = p(t=0) + dx(t=1)
# in the 2nd second his position is equal to his position at time t=2 + displacemet that 
# took place at 2nd second
# p(t=2) = p(t=1) + dx(t=2)
# but p(t=1) = p(t=0) + dx(t=1)
# so p(t=2) = p(t=0) + dx(t=1) + dx(t=2)
# likewise at third second
# p(t=3) = p(t=0) + dx(t=1) + dx(t=2) + dx(t=3)
# a common pattern emerges

# displacement at nth second is equal to the position at 0th second (start) plus the displacements
# that has occured upto nth second
# p(t=n) = p(t=0) + dx(t=1) + dx(t=2) + ....... + dx(t=n)

import numpy as np
import matplotlib.pyplot as plt
import gc
import os
os.chdir("C:\\Users\\Anoba\\Documents\\Python\\Using_Python_for_Research")

disp = np.random.normal(0,1,(100,2))  # 100x2 2 dimensional array
# first column - x coordinate
# second column - y coordinate

#plt.plot(disp[:,0], disp[:,1])
#plt.plot(disp[:,0], disp[:,1], color="green", marker="o", markersize=2.5)

cumulative_disp = np.cumsum(disp, axis=1)
cumulative_disp

#plt.plot(disp[:,0], disp[:,1], color="green", marker="o", markersize=5)
#plt.plot(cumulative_disp[:,0], cumulative_disp[:,1], color="blue", marker="o", markersize=5)


import os
#os.getcwd() # 'C:\\Program Files\\Python39'
#os.chdir("C:\\Users\\Anoba\\Documents\\Python\\Using_Python_for_Research")
#os.getcwd() # 'C:\\Users\\Anoba\\Documents\\Python\\Using_Python_for_Research'
disp = np.random.normal(0,1,(100,2))
cumulative_disp = np.cumsum(disp, axis=1)
plt.plot(disp[:,0], disp[:,1], color="green", marker="o", markersize=5)
plt.plot(cumulative_disp[:,0], cumulative_disp[:,1], color="red", marker="o", markersize=5)
plt.savefig("RandomWalks.jpeg", dpi=2500)
plt.close("all")
gc.collect()

# to get the points spread broadly rather than cluttering closer
disp = np.random.normal(2.5,10,(100,2)) 
cumulative_disp = np.cumsum(disp, axis=1)
plt.plot(disp[:,0], disp[:,1], color="green", marker="o", markersize=5)
plt.plot(cumulative_disp[:,0], cumulative_disp[:,1], color="red", marker="o", markersize=5)
plt.savefig("Random_wide_Walks.jpeg", dpi=3500)
plt.close("all")
gc.collect()

disp = np.random.normal(2.5,10,(10,2)) 
cumulative_disp = np.cumsum(disp, axis=1)
plt.plot(disp[:,0], disp[:,1], color="green", marker="o", markersize=5)
plt.plot(cumulative_disp[:,0], cumulative_disp[:,1], color="red", marker="o", markersize=5)
plt.savefig("Random_trods.jpeg", dpi=3500)
plt.close("all")
gc.collect()

# however all these plots do not start from the origin i.e (0,0)
# to do this the first row of the matrix needs to be modified
# cannot just modify the existing values but need to insert a row before it!!
# two numpy arrays can be concatanated using the .concatanate() function
# first argument takes a tuple containing the arrays to be concatanated
# second argument is the axis

# to generate the first row
disp = np.random.normal(0,1,(10,2))
cumulative_disp = np.cumsum(disp, axis=1) 
disp_0 = np.array([[0,0]])
displacement = np.concatenate((disp_0, cumulative_disp), axis=0)
plt.plot(displacement[:,0], displacement[:,1], color="blue", marker="o", markersize=5)
plt.savefig("Start_origin.jpeg", dpi=3500)
plt.close("all")
gc.collect()

# say the walker takes a 100 steps
cumulative_disp = np.cumsum(np.random.normal(0,1,(100,2)), axis=1) 
displacement = np.concatenate((np.array([[0,0]]), cumulative_disp), axis=0)
plt.plot(displacement[:,0], displacement[:,1], color="purple", marker="o", markersize=5)
plt.savefig("100_steps.jpeg", dpi=3500)
plt.close("all")
gc.collect()

cumulative_disp = np.cumsum(np.random.normal(0,1,(2,100)), axis=1)  # changing the dimension of the matrix doesn't change anything
displacement = np.concatenate((np.array([[0],[0]]), cumulative_disp), axis=1)
plt.plot(displacement[0], displacement[1], color="red", marker="o", markersize=5)
plt.savefig("100_steps1.jpeg", dpi=3500)
plt.close("all")
gc.collect()

cumulative_disp = np.cumsum(np.random.normal(0,1,(2,100000)), axis=1)  
displacement = np.concatenate((np.array([[0],[0]]), cumulative_disp), axis=1)
plt.plot(displacement[0], displacement[1], color="red", marker="o", markersize=5)
plt.savefig("100000_steps.jpeg", dpi=3500)
plt.close("all")
gc.collect()