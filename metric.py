import numpy as np

class Metric:
    def kld(self, var, p, q):
        """
            Calculate the Kullback-Lieber Divergence (semi-metric)
            var: the random variable to be analyzed.
            p and q: two different distributions over var with same scope.
            return: distance between p and q.
        """
        if len(p) != len(q):
            return 0
        quotient = np.divide(p, q)
        #print(":)))", len(p))
        #print("dist1:",p)
        #print("dist2",q)
        #print(":(((", len(q))
        log2_quotient = np.log2(quotient)
        divergence = np.dot(p, log2_quotient)
        return divergence

    def jsd(self, var, p, q):
         """
            Calculate the Jensen-Shannon Divergence (true metric)
            var: the random variable to be analyzed.
            p and q: two different distributions over var with same scope.
            return: distance between p and q.
         """
         m = 1./2. * (p + q)
         return 1./2. * (self.kld(var, p, m) + self.kld(var, q, m))

    def mid(self, p, q):
        """
            Calculates distances between two probability distribution represented by
            Chow Liu Trees by means of the Mutual Information amongst their variables.
            p and q: the Chow Liu Trees to compare.
            return: the average percentage of similarity between p and q.
        """
        # get the absolute differences between both distributions' mutual information vectors
        differences = abs(p.mi_vec - q.mi_vec) # each vector contains the MI's between all variables
        # express the distances in terms of similiarities (to be used as percentages of similarity)
        similarities = np.array([1 - similarity for similarity in differences])
        # return the average percentage of similarity
        #print('---')
        #print(similarities)
        similarity = 1.0 * np.sum(similarities) / len(similarities)
        return similarity

    def divergence(self, clt1, clt2, metric):
        """
            Calculates divergence between probability distributions represented as Chow-Liu Trees
            clt1 and clt2: are the two Chow-Liu Trees to be compared.
            return: the average percentage of divergence between distributions of clt1 and clt2.
        """
        divergences = []
        for var, _ in clt1.X.items():
            # get distibutions for the current variable
            distribution_1 = clt1.probability_distribution(var)
            distribution_2 = clt2.probability_distribution(var)
            # choose a metric
            if metric == "kld": # kullback-lieber
                divergence = self.kld(var, distribution_1, distribution_2)
            elif metric == "jsd": # jensen-shannon
                divergence = self.jsd(var, distribution_1, distribution_2)
            elif metric == "mid": # mutual information distance
                return self.mid(clt1, clt2)
            # store each result as a percentage of similarity
            divergences.append(1 - divergence)
        # return the average divergence
        return 1.0 * np.sum(divergences) / len(divergences)

    def conditional_probability_tests(self, clt1, clt2, which="all"):
        """
            Runs conditional probability queries for the clt1 and clt2 and comparing results.
            clt1 and clt2: the Chow Liu Trees for running the conditional probability test.
            which: determines wether to run specific queries or all possible conbinations. 
            return: average error in the query results expressed in percentages.
        """
        if which == "all":
            # arrays for storing the results of inferences for each clt
            clt1_inferences = np.array([])
            clt2_inferences = np.array([])
            # get all unique values from the distributions
            values = set([val for _, vals in clt1.X.items() for val in vals])
            variables = sorted(list(clt1.X.keys()))
            for i in range(0, len(variables) - 1):
                current = variables[i + 1]
                parent = variables[i]
                for value_cur in values:
                    for value_par in values:
                        clt1_inferences = np.append(clt1_inferences, clt1.conditional_probability(current, parent, value_cur, value_par))
                        clt2_inferences = np.append(clt2_inferences, clt2.conditional_probability(current, parent, value_cur, value_par))
        elif which == "specific":
            # this will only work for the toy set
            clt1_inferences = np.append(clt1_inferences, clt1.conditional_probability('1', '0', 'A', 'A'))
            clt1_inferences = np.append(clt1_inferences, clt1.conditional_probability('1', '0', 'C', 'A'))
            clt1_inferences = np.append(clt1_inferences, clt1.conditional_probability('1', '0', 'A', 'G'))
            clt1_inferences = np.append(clt1_inferences, clt1.conditional_probability('1', '0', 'C', 'G'))

            clt2_inferences = np.append(clt2_inferences, clt2.conditional_probability('1', '0', 'A', 'A'))
            clt2_inferences = np.append(clt2_inferences, clt2.conditional_probability('1', '0', 'C', 'A'))
            clt2_inferences = np.append(clt2_inferences, clt2.conditional_probability('1', '0', 'A', 'G'))
            clt2_inferences = np.append(clt2_inferences, clt2.conditional_probability('1', '0', 'C', 'G'))

        return self.query_differences(clt1_inferences, clt2_inferences)

    def query_differences(self, clt1_inferences, clt2_inferences):
        """
            Calculates the distance between the results of queries of two CLT.
            clt1_inferences and clt2_inferences: inference results from clt1 and clt2 respectively.
            return: average error in the query results.
        """
        differences =  abs(clt1_inferences - clt2_inferences)
        similarities = np.array([1 - difference for difference in differences])
        error = 1.0 * np.sum(similarities) / len(similarities)
        return np.sum(error)