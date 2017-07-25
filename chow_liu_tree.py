"""
Chow-Liu Algorithm implemenation for building CL Tress and making inferences.

Author: Jose Gonzalez
Date: July 2017
"""
import numpy as np
import networkx as nx
import matplotlib.pyplot as plt

class Chow_Liu_Tree:
    def __init__(self, X=None, mi_vec=np.array([]), testing=False):
        """
            Initialization function
            X: data set.
            mi_vec: mutual information vector with MI values between all variables.
            testing: testing flag.
        """
        self.X = X
        self.mi_vec = mi_vec
        self.testing = testing
        self.tree = nx.DiGraph() 

    def probability_distribution(self, var):
        """
            Generates the probability distribution for a given variable
            var: the variable to be analyzed.
            return: a vector with the probability of each possible value.
        """
        # get a list of all the possible values for the given variable
        values = [val for _, vals in self.X.items() for val in vals]
        distribution = []
        for value in set(values):
            # calculate the marginals
            mp = self.marginal_probability(var, value)
            #print("for", var, "=", value, "prob:", mp)
            # ignore impossible values for var
            if mp != 0:
                distribution.append(mp)
        return np.array(distribution)

    def conditional_probability(self, var, par, val_var, val_par):
        """
            Calculate a conditional probability.
            var: the variable in which the inference will be done.
            par: the parent of var.
            val_var: the value of the current variable.
            val_par: the value of the parent variable of val_var.
            return: p(var|par).
        """
        try:
            return 1.0 * self.joint_probability(var, par, val_var, val_par) / self.marginal_probability(par, val_par)
        except:
            return 0

    def calculate_mutual_information(self):
        """ 
            Calculate mutual information between variables.
            return: list with MI values in the fully conneted graph.
        """
        m_info_list = [] 
        for var1 in range(len(self.X)):
            for var2 in range(var1):
                m_info_list.append((str(var2), str(var1), self.mutual_information(str(var2), str(var1))))
        return m_info_list

    def build_clt(self):
        """
            Build a Chow-Liu Tree from data.
            return: tree structure.
        """
        m_info_list = self.calculate_mutual_information() # get the list of MI between all variables
        m_info_list.sort(key=lambda tup : tup[2], reverse=True) # sort mutual information list in descending order
        inserted_nodes = [] # keep track of the nodes already inserted as to not duplicate paths
        for tup in m_info_list:
            # if some node has not been inserted (the path between those nodes does not exist)
            if not (tup[0] in inserted_nodes and tup[1] in inserted_nodes):
                # build the path
                self.tree.add_node(tup[0])
                self.tree.add_node(tup[1])
                # avoid more than one parent by controlling the direction of the edges through the order of assignments
                parent, child = self.choose_parent(tup)
                self.tree.add_edge(parent, child, weight = -1 * tup[2])
                # register both nodes 
                inserted_nodes.append(tup[0])
                inserted_nodes.append(tup[1])
        return self.tree

    def choose_parent(self, tup):
        """
            Chooses between two connected variables which will become the parent on the CTL.
            tup: is a tuple with the two variables to analyze.
            return: parent and child. 
        """
        if len(self.tree.predecessors(tup[1])) != 0: # if the second node has already a parent
            # then it becomes a parent
            return tup[1], tup[0]
        else: # otherwise the first node is assigned as the parent
            return tup[0], tup[1]

    def plot(self):
        """
            Plot the CLT and wait.
        """
        nx.draw(self.tree, with_labels=True)
        plt.draw()
        plt.waitforbuttonpress()

    def mutual_information(self, var1, var2):
        """
            Compute mutual information between two variables.
            var1 and var2: random variables to anaylze.
            return: mutual information between var1 and var2.
        """
        mutual_info = 0 # information cummulative value
        compared_vals = [] # list for filtering repeated comparisons
        for x_i in self.X[var1]: # go through all values of the first and second variable
            for x_j in self.X[var2]:
                if not ((x_i, x_j) in compared_vals): # avoid already compared values
                    jp = self.joint_probability(var1, var2, x_i, x_j)
                    mp_v1 = self.marginal_probability(var1, x_i)
                    mp_v2 = self.marginal_probability(var2, x_j)
                    # jp != 0 for filtering only values that belong to the distribution
                    mutual_info += 1.0 * jp * np.log2(1.0 * jp / (mp_v1 * mp_v2)) if (jp != 0) else 0
                    compared_vals.append((x_i, x_j))
        self.mi_vec = np.append(self.mi_vec, mutual_info)
        return mutual_info

    def marginal_probability(self, var, x):
        """
            Compute the marginal of a variable for a specific value.
            var: name of the variable.
            x: specific probable value.
            return: probability of var having value x.
        """
        return 1.0 * self.X[var].count(x) / len(self.X[var])

    def joint_probability(self, var1, var2, x_i, x_j):
        """
            Compute the joint probability of two variables given one possible value for each variable.
            X: data set.
            var1: first variable.
            var2: second variable.
            x_i: value for the first variable.
            x_j: value for the second variable.
            return: joint probability of var1 and var2 for values x_i and x_j. 
        """
        var1_events = self.X[var1] # event list from data for variable 1
        var2_events = self.X[var2] # event list from data for variable 2
        appearances = 0 # counter for the occurances of a particular combination event list data
        combination = (x_i, x_j) # set the combination with the values given
        for tup in zip(var1_events, var2_events): # go through the event list of both variables
            if tup == combination: # if the specified combination is found then increment the counter
                appearances += 1
        return 1.0 * appearances / len(self.X[var1]) # return the probability of that joint event

