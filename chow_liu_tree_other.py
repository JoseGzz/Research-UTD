#
# Copyright John Reid 2011
#
import numpy as N, networkx as nx
import matplotlib.pyplot as plt
from collections import defaultdict

def marginal_distribution(X, u):
    """
    Return the marginal distribution for the u'th features of the data points, X.
    """
    values = defaultdict(float)
    s = 1. / len(X)
    for x in X:
        values[x[u]] += s
    return values

def marginal_pair_distribution(X, u, v):
    """
    Return the marginal distribution for the u'th and v'th features of the data points, X.
    """
    if u > v:
        u, v = v, u
    values = defaultdict(float)
    s = 1. / len(X)
    for x in X:
        values[(x[u], x[v])] += s
    return values

def calculate_mutual_information(X, u, v):
    """
    X are the data points.
    u and v are the indices of the features to calculate the mutual information for.
    """
    if u > v:
        u, v = v, u
    marginal_u = marginal_distribution(X, u)
    marginal_v = marginal_distribution(X, v)
    marginal_uv = marginal_pair_distribution(X, u, v)
    I = 0.
    for x_u, p_x_u in marginal_u.iteritems():
        for x_v, p_x_v in marginal_v.iteritems():
            if (x_u, x_v) in marginal_uv:
                p_x_uv = marginal_uv[(x_u, x_v)]
                I += p_x_uv * (N.log(p_x_uv) - N.log(p_x_u) - N.log(p_x_v))
    return I

def build_chow_liu_tree(X, n):
    """
    Build a Chow-Liu tree from the data, X. n is the number of features. The weight on each edge is
    the negative of the mutual information between those features. The tree is returned as a networkx
    object.
    """
    print("hello world")
    G = nx.DiGraph()
    for v in xrange(n):
        G.add_node(v)
        for u in xrange(v):
            print("Training...")
            mi = calculate_mutual_information(X, u, v)
            print("mi:" + str(v) + " and " + str(u) + " is "  + str(mi))
            G.add_edge(u, v, weight=-mi)
    G = nx.minimum_spanning_tree(G)
    nx.draw(G)
    #nx.draw(G,pos=nx.spring_layout(G))
    plt.draw()
    plt.pause(100)
    print("done")
    #return T

#if '__main__' == __name__:
#    import doctest
#doctest.testmod()
X = [
'AACC',
'AAGC',
'AAGC',
'GCTC',
'ACTC',
]
print(marginal_pair_distribution(X, 0, 1))
#build_chow_liu_tree(X, len(X[0]))
