from chow_liu_tree import Chow_Liu_Tree 
import matplotlib.pyplot as plt
import numpy as np
import random
from copy import deepcopy as dc
from metric import Metric

def perturbate_distribution(X, X2):
    X_aux = X2
    while(True):
        variable = random.choice(X2.keys())
        new_val_index = random.choice(range(len(X2[variable]) - 1))
        current_val = X2[variable][new_val_index]
        X2[variable][new_val_index] = random.choice(X2[variable])
        if set(X[variable]) == set(X2[variable]):
            return X2
        else:
            X2[variable][new_val_index] = current_val


def split_data(X):
    return np.split(X, 2)

if __name__ == "__main__":
    split_testing = False
    file_txt = np.loadtxt('data/abalone.test.data', delimiter=',')
    #print(file_txt)
    #import sys
    #sys.exit()

    data = []
    for line in file_txt.T:
        d_aux = []
        for num in line:
            d_aux.append(str(num))
        data.append(d_aux)
    
    points_stack = []

    if split_testing:
        data1_full, data2_full = split_data(file_txt)
        for point_d1, point_d2 in zip(data1_full[::-1], data2_full[::-1]):
            points_stack.append((point_d1, point_d2))
        points = points_stack.pop()
        new_point1, new_point2 = points[0], points[1]
        X = {str(i) : [str(e)] for i, e in enumerate(new_point1)}
        X2 = {str(i) : [str(e)] for i, e in enumerate(new_point2)}
    else:
        X = {str(i) : e for i, e in enumerate(data)}
        X2 = dc(X)

    #print(X)
    #print(X2)
    #import sys
    #sys.exit()
    

    """
    X = {
     '0': ['A','A','A','G','A'],
     '1': ['A','A','A','C','C'],
     '2': ['C','G','G','T','T'],
     '3': ['C','C','C','C','C'],
    }

    X2 = {
     '0': ['A','A','A','G','A'],    
     '1': ['A','A','A','C','C'],
     '2': ['C','G','G','T','T'],
     '3': ['C','C','C','C','C'],
    }
    """

    fig = plt.figure()
    divergences = []
    divergences2 = []
    divergences3 = []
    metric = Metric()
    iterations = len(points_stack) if split_testing else 500
    for i in range(iterations):
        
        values = [val for _, vals in X.items() for val in vals]
        clt = Chow_Liu_Tree(X)
        clt2 = Chow_Liu_Tree(X2)
        
        clt.build_clt()
        clt2.build_clt()
        
        structure_diff = 100 * round(metric.divergence(clt, clt2, "mid"), 7)
        inference_diff = 100 * round(metric.conditional_probability_tests(clt, clt2), 7)
        #print('hi')
        #print("clt and clt2 are: ", str(structure_diff) + "% ", "the same structure.")
        #print("clt and clt2 are: ", str(inference_diff) + "% ", "the same at inference.")
        #divergences.append((structure_diff, inference_diff))
        divergences.append((inference_diff, 100 * metric.divergence(clt, clt2, "jsd")))
        divergences2.append((inference_diff, 100 * metric.divergence(clt, clt2, "kld")))
        divergences3.append((inference_diff, structure_diff))
        # compute marginals, joints and conditionals for testing
        
        if split_testing:
            points = points_stack.pop()
            new_point1, new_point2 = points[0], points[1]
            for i, val in enumerate(new_point1):
                X[str(i)].append(str(val))
            for i, val in enumerate(new_point2):
                X2[str(i)].append(str(val))
        else:
            X2 = perturbate_distribution(X, X2)
        #print(X)
        #print(X2)
    
    #print(divergences)
    #plt.xlabel("Structure")
    #plt.xlabel(metric_name)
    #plt.scatter(*zip(*divergences))
    #clt.plot()
    #clt2.plot()
    #print(X)
    #print(X2)
    #clt2.plot()
    """
    fig1 = plt.figure()
    ax1 = fig1.add_subplot(111)
    ax1.scatter(*zip(*divergences))
    ax1.set_xlabel('x')
    ax1.set_ylabel('y')


    fig2 = plt.figure()
    ax2 = fig2.add_subplot(111)
    ax2.scatter(*zip(*divergences2))
    """
    plt.figure(0)
    plt.xlabel("Inference")
    plt.ylabel("jsd")
    plt.scatter(*zip(*divergences))

    plt.figure(1)
    plt.xlabel("Inference")
    plt.ylabel("kld")
    plt.scatter(*zip(*divergences2))

    plt.figure(2)
    plt.xlabel("Inference")
    plt.ylabel("MI")
    plt.scatter(*zip(*divergences3))
    
    plt.show()

    if clt.testing:
        print("marginal of 0 = A: " + str(clt.marginal_probability('0', 'A')))
        print("marginal of 0 = G: " + str(clt.marginal_probability('0', 'G')))
        print("marginal of 1 = A: " + str(clt.marginal_probability('1', 'A')))
        print("marginal of 1 = C: " + str(clt.marginal_probability('1', 'C')))
        print("marginal of 2 = C: " + str(clt.marginal_probability('2', 'C')))
        print("marginal of 2 = T: " + str(clt.marginal_probability('2', 'T')))
        print("marginal of 2 = G: " + str(clt.marginal_probability('2', 'G')))
        print("marginal of 3 = C: " + str(clt.marginal_probability('3', 'C')))
        print("------")
        print("joint of 0 = A and 1 = A: " + str(clt.joint_probability('0', '1', 'A', 'A')))
        print("joint of 0 = G and 1 = C: " + str(clt.joint_probability('0', '1', 'G', 'C')))
        print("joint of 0 = A and 1 = C: " + str(clt.joint_probability('0', '1', 'A', 'C')))
        print("joint of 1 = A and 2 = A: " + str(clt.joint_probability('1', '2', 'A', 'G')))
        print("joint of 1 = G and 2 = C: " + str(clt.joint_probability('1', '2', 'C', 'T')))
        print("joint of 1 = A and 2 = C: " + str(clt.joint_probability('1', '2', 'A', 'C')))
        print("------")
        for v in range(len(X)):
            for u in range(v):
                print("mi " + str(u) + " and " + str(v) + ": " + str(clt.mutual_information(str(u), str(v))))
        print('------')
        print(clt.conditional_probability('1', '0', 'A', 'A'))
        print(clt.conditional_probability('1', '0', 'C', 'A'))
        print('----')
        print(clt.conditional_probability('1', '0', 'A', 'G'))
        print(clt.conditional_probability('1', '0', 'C', 'G'))
        print("clt2")
        print('------')
        print(clt2.conditional_probability('1', '0', 'A', 'A'))
        print(clt2.conditional_probability('1', '0', 'C', 'A'))
        print('----')
        print(clt2.conditional_probability('1', '0', 'A', 'G'))
        print(clt2.conditional_probability('1', '0', 'C', 'G'))