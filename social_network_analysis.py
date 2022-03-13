# introduction to network analysis
# many systems consist of large number of interacting components
# these structures can be represented as networks where the network nodes represent the components
# network edges represent the interactions between the componenets
# network analysis can be used to study how pathogens, behaviours and information spread
# in social networks

# network is the real world object like road networks, lake networks, railway networks
# graphs are the abstract mathematical representation of networks
# graphs contain nodes(vertices), edges(links)
# mathematically a graph is a collection of vertices and edges where each edge corresponds to 
# a pair of vertices
# in standard visualization practices vertices are drawn as circles and the edges as lines connecting the circles

# two connected vertices are called neighbours
# the degree of a vertex is the number of entries(other vertices) connected to it
# path is a sequence of unique vertices
# length of a path is the number of edges that make up that path
# if a vertex has a valid path to another vertex, these vertices are considered reachable
# a graph is considered connected if every vertex in that graph is reachable from every vertex in that graph
# if not the graph is considered disconnected

# components are a subset of graphs, made of few interconnected vertices that
# are isolated from other components in the graph
# size of a component is defined as the number of nodes present in that component
# if a graph contains multiple components, the largest component (with the highest number of nodes)
# called the largest connected component


# basics of NetworkX module
import networkx as nx

# creating a network of undirected graph using the graph() function
g = nx.Graph()

# nodes can be added to this graph one/several at a time
g.add_node(1)  # 1 is a label of a node, i.e node number one NOT THE NUMBER OF NODES

# to add multiple nodes to a graph
g.add_nodes_from([2,3,4,5]) 

# node labels do not have to be numbers
g.add_nodes_from(["A","B","C"])

# to see all the nodes in the graph
g.nodes()
# NodeView((1, 2, 3, 4, 5, 'A', 'B', 'C'))

# similar functions exist to add edges
# edges are treated as pair of nodes
g.add_edge(1,"A") # this edge connects the nodes 1 and A
g.add_edge("A","C")

# add multiple adges simultaneously
g.add_edges_from([("B","C"),(2,4),(4,1),(4,"A"),(3,2)])
# pass 1/more tuples inside a list, each tuple representing an edge

# edges can be created for nodes that did not exist at the time of edge creation
# here Python creates the missing node and connects it using the edge

g.add_edges_from([("B","D"),(0,3),(0,2),(0,"A")])
# nodes D and 0 didn't exist before

# to see all the edges in the graph
g.edges()
# EdgeView([(1,'A'), (1,4), (2,4), (2,3), (2,0), (3,0), (4,'A'), ('A','C'), ('A',0), ('B','C'), ('B','D')])
g.nodes()
# NodeView((1, 2, 4, 5, 'A', 'B', 'C', '3', ',', 'D', 0))

# nodes and edges can be removed from graphs
g.remove_node(3)
g.nodes()
# NodeView((1, 2, 4, 5, 'A', 'B', 'C', 'D', 0))
# node 3 has been removed

# remove multiple nodes
g.remove_nodes_from(["D",2,1])
g.nodes()
# NodeView((4, 5, 'A', 'B', 'C', 0))

# analogous to node removal, edges can also be removed from graphs
g.edges()
# EdgeView([(4, 'A'), ('A', 'C'), ('A', 0), ('B', 'C')])
g.remove_edge(4,"A")
g.edges()
# EdgeView([('A', 'C'), ('A', 0), ('B', 'C')])
# removing multiple edges at once
g.remove_edges_from([("A","C"),("A",0)])
g.edges()
# EdgeView([('B', 'C')])

# to find the number of nodes and edges in graphs
g.number_of_nodes()  # 6
g.number_of_edges()  # 1

G = nx.Graph()
G.add_nodes_from([1,2,3,4])
G.add_edges_from([(1,2),(3,4)])
G.number_of_nodes(), G.number_of_edges()  # 4,2


# graph visualizations
# NetworkX library packs some grapghs with itself
# a karate club graph that contains club members as nodes and friendships as edges
karate = nx.karate_club_graph()

# NetworkX isn't really made for visualizations but it can be used to create some basic visualizations

import matplotlib.pyplot as plt
# NetworkX depends on matplotlib for plotting, thus matplotlib module needs to be imported
nx.draw(karate, with_labels=True, edge_color="#020202", node_color="#93FFDD", node_size=900)
plt.show()  # to view the graph plt.show() is essential

nx.draw(karate, with_labels=True, edge_color="#7F88FF", node_color="#FFF67F", node_size=900)
plt.savefig("karate.jpeg", dpi=1000)

# networkx stores degrees of nodes in a dictionary where the keys are node labels and the values are
# their associated degrees
karate.degree()
# DegreeView({0: 16, 1: 9, 2: 10, 3: 6, 4: 3, 5: 4, 6: 4, 7: 4, 8: 5, 9: 2, 10: 3, 11: 1, 12: 2, 13: 5, 14: 2, 15: 2, 16: 2, 17: 2, 18: 2, 19: 3, 20: 2, 21: 2, 22: 2, 23: 5, 24: 3, 25: 3, 26: 2, 27: 4, 28: 3, 29: 4, 30: 4, 31: 6, 32: 12, 33: 17})

karate.degree()[33]  # 17 using Python's dictionary lookup method
karate.degree(33)  # 17 using networkx's own query method

karate.degree(33) is karate.degree()[33]  # True
karate.nodes() 
len(karate.nodes())  # 34
karate.edges()
len(karate.edges())  # 78

# random graphs
# similar to generating random numbers from a given distribution, say normal distribution/binominal distribution et.c
# random graphs can be sampled from an ensemble of random graphs
# similar to different distributions of numbers, there are different random graph models that generate different 
# random graphs
# simplest random graph model is Erdos-Renyi (ER) model
# this family of random graphs have 2 parameters
# N - number of nodes in graph
# p - probability for any pair of nodes to be connected by an edge


# eventhough networkx module includes a random ER graph generator
# lets define a function

from scipy.stats import bernoulli
# .rvs() returns random variate/s of specified type
# there are just 2 possibilities; nodes being conneceted or disconnected
# connected = 1
# disconnected = 0
# assign the probability of success (being connected) as 0.2 (example)
bernoulli.rvs(p=0.2)
# returns 0s most of the time, as expected


# defining a function to generate random ER graphs for given N & p
# pseudocode

# create an empty graph
# add N number of nodes
# loop over all nodes
    # add edges with a probability of p

N = 50
p=0.1

er = nx.Graph()
er.add_nodes_from(list(range(N))) # N number of nodes with labels ranging from 0 to N-1
for n1 in er.nodes():
    for n2 in er.nodes():  # edge is a pair of nodes, thus two for loops are nested
        if bernoulli.rvs(p=p) == True:  
            er.add_edge(n1,n2)
import matplotlib.pyplot as plt
nx.draw(er, node_color="#C0FC77", edge_color="#F16317", with_labels=True, node_size=800)
plt.show()
  
# in if statements, the result to proceed is a True/1 it doesn't need to be stated explicitly
# for example; if bernoulli.rvs(p=p):  will work just fine
    
# the above graph has edges connecting the same nodes
# and is too dense for an ER graph with p=0.1
# this is because each node is considered twice as two for loops are nested
# eg. there will be two instances for (n1,n2) and (n2,n1) for each node
# since this graph isn't direectional (n1,n2) is identical to (n2,n1)
# code need to be modified!!!

# if n1<n2 or n1>n2 will force the programme to consider each pair of nodes just once!!

N = 50
p = 0.2
er = nx.Graph()
# N number of nodes with labels ranging from 0 to N-1
er.add_nodes_from(list(range(N)))
for n1 in er.nodes():
    for n2 in er.nodes():  # edge is a pair of nodes, thus two for loops are nested
        if bernoulli.rvs(p=p) == True and n1 < n2:
            er.add_edge(n1, n2)
nx.draw(er, node_color="#C0FC77", edge_color="#F16317", with_labels=True, node_size=800)
plt.show()

def er_graph(N,p=0.2):
    """
    Takes the number of nodes as N and the probability of a pair of nodes to be connected as p
    and returns a Erdos-Reyni graph.
    """
    er = nx.Graph()
    er.add_nodes_from(list(range(N)))
    for n1 in er.nodes():
        for n2 in er.nodes():  # edge is a pair of nodes, thus two for loops are nested
            if bernoulli.rvs(p=p) == True and n1 < n2:
                er.add_edge(n1, n2)
    nx.draw(er, node_color="#C0FC77", edge_color="#F16317",
        with_labels=True, node_size=800)
    plt.show()

er_graph(100,0.05)
er_graph(20,.2)
er_graph(1000,0.05)

er_graph(10,1)
# here the probability for any pair of nodes to be connected is 1
# meaning all nodes will be connected, no isolated nodes or components

er_graph(10,0)
# here the probability for any pair of nodes to be connected is 0
# meaning all nodes will be disconnected, all nodes will be isolated and can be considered as separate components

# plotting the degree distribution


def er_graph(N, p=0.2):
    er = nx.Graph()
    er.add_nodes_from(list(range(N)))
    for n1 in er.nodes():
        for n2 in er.nodes():  # edge is a pair of nodes, thus two for loops are nested
            if bernoulli.rvs(p=p) == True and n1 < n2:
                er.add_edge(n1, n2)
    return er

def plot_degree_distribution(g):
    x = dict(g.degree()).values() # extracts the degrees of each node
    plt.hist(x, histtype="step", stacked=True, color="#E773FB", linewidth=2)
    plt.xlabel("Degree $k$", fontsize=15)
    plt.ylabel("P(k)", fontsize=15)
    plt.title("Degree distribution", fontsize=20)
    plt.show()

g = er_graph(50, 0.08)
plot_degree_distribution(g)

g = er_graph(500, 0.08)
plot_degree_distribution(g)

D = {1: 1, 2: 2, 3: 3}
plt.hist(D)
plt.show()

g1 = nx.erdos_renyi_graph(100, 0.03)
nx.draw(g1, node_color="#C0FC77", edge_color="#F16317", with_labels=True, node_size=800)
plt.show()

g2 = nx.erdos_renyi_graph(100, 0.30)
nx.draw(g2, node_color="#C0FC77", edge_color="#F16317", with_labels=True, node_size=800)
plt.show()

plt.figure(figsize=(20,20))
plt.rcParams["font.size"]=15
plt.subplot(121)
plt.hist(dict(g1.degree()).values(), histtype="step", stacked=True, color="#F5DB32", linewidth=2)
plt.xlabel("Degrees of nodes")
plt.ylabel("Frequency")
plt.title("N=500, p=0.03")
plt.subplot(122)
plt.hist(dict(g2.degree()).values(), histtype="step", stacked=True, color="#4CF532", linewidth=2)
plt.title("N=500, p=0.30")
plt.xlabel("Degrees of nodes")
plt.ylabel("Frequency")
plt.show()

# descriptive statistics of empirical social networks
# social networks from 2 different villages in rural India; don't confuse this with social media
# these data are part of a microfinance study
# the data contains the details about the relationships the villagers have with others in the village
# structure of connections in a network is captured in a matrix, called the adjacency matrix of the network
# for a network with N nodes, its adjacency matrix is an NxN matrix
# entry (i,j) will be 1 if the nodes i & j are connected with an edge, if not entry (i,j) will be 0
# these graphs are undirected, since the entry (i,j) is the same as entry (j,i) just means that these 2 nodes are connected
# the edge is not directional
# due to this the adjacency matrix is symmetric

# use numpy.loadtxt() to read in .csv files
import numpy as np
matrix_1 = np.loadtxt("C:\\Users\\Anoba\\Documents\\Python\\Using_Python_for_Research\\adj_allVillageRelationships_vilno_1.csv", delimiter=",", dtype=int)
matrix_2 = np.loadtxt("C:\\Users\\Anoba\\Documents\\Python\\Using_Python_for_Research\\adj_allVillageRelationships_vilno_2.csv", delimiter=",", dtype=int)

# since it is a text file reading function, the delimiter needs to be passed explicitly
# since the ones and zeros here are boolean values (True/False) they can be loaded in as integers for convenience

# creating graph objects using the adjacency matrices
graph_1 = nx.to_networkx_graph(matrix_1)
graph_2 = nx.to_networkx_graph(matrix_2)

nx.draw(graph_1, edge_color="black", node_size=900, node_color="yellow", with_labels=True)
nx.draw(graph_2, edge_color="black", node_size=900, node_color="yellow", with_labels=True)
# both graphs are too cluttered to make any meaningful inferences

graph_1.nodes()  # 843 nodes
graph_2.nodes()  # 877 nodes

# converting degreeview objects to standard python dictionaries
g1_deg = dict(graph_1.degree())
g2_deg = dict(graph_2.degree())

plt.figure(figsize=(40,25))
plt.rcParams["font.size"]=15
plt.subplot(121)
plt.hist(g1_deg.values(), histtype="bar", color="green", label="Village 1", linewidth=2.5, bins=np.linspace(0,55,56))
plt.xlabel("Degrees")
plt.ylabel("Distribution of degrees")
plt.legend(loc="best", fontsize=15)
plt.subplot(122)
plt.hist(g2_deg.values(), histtype="bar", color="red", label="Village 2", linewidth=2.5, bins=np.linspace(0,35,36))
plt.xlabel("Degrees")
plt.ylabel("Distribution of degrees")
plt.legend(loc="best", fontsize=15)
plt.show()

# to compute the number of nodes, edges and mean number of degrees for graphs
def graph_stats(g):
    """Takes in a networkX graph object and returns its summary statistics"""
    print("Number of nodes in the graph:", " = ", g.number_of_nodes())
    print("Number of edges in the graph:", " = ", g.number_of_edges())
    print("Mean degrees of graph:",  " = ", np.mean(np.mean(list(dict(g.degree()).values()))))


graph_stats(graph_1)
# Number of nodes in the graph:  =  843
# Number of edges in the graph:  =  3405
# Mean degrees of graph:  =  8.078291814946619
graph_stats(graph_2)
# Number of nodes in the graph:  =  877
# Number of edges in the graph:  =  3063
# Mean degrees of graph:  =  6.985176738882554

# finding the largest connected component
# extracting connected components
# nx.connected_components() extracts the connected components from the graphs
c_comp_1 = nx.connected_components(graph_1)
c_comp_2 = nx.connected_components(graph_2)

# nx.connected_components() is a generator function; which doesn't return an output
# generator functions can be used to generate a sequence of objects using the .__next__() method!!
# in the case of nx.connected_components(), .__next__() will generate a component object

c_comp_1.__next__()
# {0, 1, 2, 3,.......... 839, 840, 841, 842}
# contains all nodes of graph_1

# generator objects can only be used once in iteration
# any iteration such as list comprehensions/loops will exhaust the generator objects, they can't be used again
# when exhausted generator objects are called a stopiteration error will be returned!

# creating subgraphs for connected components
# networkx graph objects generate subgraphs for each component in the networkx connected components object!.
cc_sgraph_1 = [graph_1.subgraph(x) for x in nx.connected_components(graph_1)]
cc_sgraph_2 = [graph_2.subgraph(x) for x in nx.connected_components(graph_2)]

a = cc_sgraph_1[0]  # voila
type(a)  # networkx.classes.graph.Graph
a.number_of_nodes()  # 825  !! graph_1 has a total of 843 nodes.
a.number_of_edges()  # 3386 !! graph_1 has a total of 3405 edges

b = cc_sgraph_2[2]
b.number_of_nodes() # 1
b.number_of_edges() # 0 an isolated component with a single node

c = cc_sgraph_2[1]
c.number_of_nodes() # 2
c.number_of_edges() # 1

d = cc_sgraph_2[0]
d.number_of_nodes() # 810  graph_2 has a total of 877 nodes
d.number_of_edges() # 2924 graph_2 has a total of 3063 edges

# extracting the largest connected components from each graph
LCC_graph_1 = max([graph_1.subgraph(comp) for comp in nx.connected_components(graph_1)], key=len)
# sorts the elements based on length and returns the element with maximum length
LCC_graph_2 = max([graph_2.subgraph(comp) for comp in nx.connected_components(graph_2)], key=len)

LCC_graph_1.number_of_nodes()  # 825
LCC_graph_1.number_of_edges()  # 3386
LCC_graph_2.number_of_nodes()  # 810
LCC_graph_2.number_of_edges()  # 2924

# proportion of nodes encompassed by LCCs
LCC_graph_1.number_of_nodes()/graph_1.number_of_nodes()  # 0.9786476868327402
LCC_graph_2.number_of_nodes()/graph_2.number_of_nodes()  # 0.9236031927023945

plt.figure(figsize=(200,100))
plt.subplot(121)
plt.title("Components of village 1")
for i in [graph_1.subgraph(comp) for comp in nx.connected_components(graph_1)]:
    nx.draw(i, with_labels=True, node_size=800)
plt.subplot(122)
plt.title("Components of village 2")
for i in [graph_2.subgraph(comp) for comp in nx.connected_components(graph_2)]:
    nx.draw(i, with_labels=True, node_size=800)
plt.show()
# works good but the graph is damn messy

plt.figure(figsize=(200,100))
plt.subplot(121)
plt.title("Largest connected component of village 1")
nx.draw(LCC_graph_1, with_labels=True, node_size=800, node_color="#C0FC77", edge_color="#F16317")
plt.subplot(122)
plt.title("Largest connected component of village 2")
nx.draw(LCC_graph_2, with_labels=True, node_size=800, node_color="#65FF7F", edge_color="#F16317")
plt.show()

plt.figure(figsize=(200,100))
plt.subplot(121)
plt.title("Largest connected component of village 1")
nx.draw(LCC_graph_1, node_size=100, node_color="green", edge_color="grey", alpha=0.4, edgecolors="black")
plt.subplot(122)
plt.title("Largest connected component of village 2")
nx.draw(LCC_graph_2, node_size=100, node_color="red", edge_color="grey", alpha=0.4, edgecolors="black")
plt.show()

# this plot shows that the LCC of village 2 has two major clusters of nodes making up the LCC
# these are called network communities

# Lists, tuples, dictionaries, and sets are all iterable objects.
# They are iterable containers which you can get an iterator from.
# an iterator is an object which implements the iterator protocol,
# which consist of the methods __iter__() and __next__()

# All iterable objects have an iter() method which is used to get an iterator
l =  (1,2,3,4,5,6,7,8,9)
L = iter(l)
print(next(L))  # 1
print(next(L))  # 2
print(next(L))  # 3
print(next(L))  # 4
print(next(L))  # 5
print(next(L))  # 6

