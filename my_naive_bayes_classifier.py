import numpy as np
import pandas as pd
from math import pi, exp
from numpy import exp, sqrt
from scipy.integrate import quad
from scipy.stats import norm

"""
Personal thought:
I was trying to implement the bayes classifier on a numerical dataset at the beginning.
But the problem with this is that the value of a numarical dataset is not that common to be the same (means 0.001 is not that common to be shown on feature 1)
So I think that is the reason why bayes is commonly used in NLP. 
"""

def gaussian_distribution(x, mean, std):
    """
    Same as: scipy.stats.norm.pdf(x, loc=mean, scale=std)
    """
    # return (1/sqrt(2*pi*std**2))*exp(-((x-mean)/std)**2/2)
    return norm.pdf(x, loc=mean, scale=std)

class IrisNaiveBayesClassifier:
    def __init__(self, path="data/iris.data", class_col="class"):
        self.dtypes = {
            'sepal_length_in_cm':np.float, 
            'sepal_width_in_cm':np.float, 
            'petal_length_in_cm':np.float, 
            'petal_width_in_cm':np.float
        }
        self.data = pd.read_csv(path,dtype=self.dtypes)
        self.class_col = class_col
        # xi_cj denote { c1 :{x1:0,...xn:0}, c2:{x1:0,...xn:0} ... cm:{x1:0,...xn:0} }
        # self.xi_cj = {cj :{} for cj in self.data[class_col].unique()}
        self.cj = {x:0 for x in self.data[class_col].unique()}

    def calculate_p_c(self, rows):
        """
        P(c)
        """
        for c in self.cj:
            self.cj[c] = len(self.data[self.data[self.class_col] == c]) / rows

    def calculate_p_x(self, x):
        """
        This is called the normalizing constant, this is actually where the classified result change 
        p(x) = p(x1)*p(x2)*p(x3)...*p(xn)
        """
        pass
        

    def calculate_p_xi_cj(self, distribution_func=gaussian_distribution, use_bin=False):
        """
        If use_bin, the continuous value will be distributed into different quartiles 

        In naive bayesian network, each feature is independent. Therefore, the P(data|class) could be separated

        This is called the hypothesis

        TODO: it is possible to implement multiprocess here since it is independent for different classes
        """
        if not use_bin:
            result = 1
            for feature in self.dtypes:
                std = self.data.groupby(self.class_col)[feature].std().reset_index(name="{}_std".format(feature))
                mean = self.data.groupby(self.class_col)[feature].mean().reset_index(name="{}_mean".format(feature))
                
                self.data = pd.merge(self.data, std, left_on=self.class_col, right_on=self.class_col)
                self.data = pd.merge(self.data, mean, left_on=self.class_col, right_on=self.class_col)

                p_feature_class = gaussian_distribution( \
                        self.data[feature], \
                        self.data["{}_mean".format(feature)], \
                        self.data["{}_std".format(feature)])

                result *= p_feature_class

                self.data.drop(["{}_mean".format(feature), "{}_std".format(feature)], axis=1, inplace=True)
            self.data["P(xi|cj)"] = result

    def fit(self):
        '''
        Bayesian thereom:
        P(data|class) * P(class) = P(class|data) * P(data)

        Assumption 1: xi are independent(otherwise, a bayesian network is used)
        P(data|cj) = P(x1|class_j) * P(x2|class_j) ... P(xn|class_j)

        P(class_j|data) = P(class_j) * P(data|class_j) / P(data)
        P(data) = SUM_j(P(data|class_j)*P(class_j))
        P(data) could be ignored since it is the same for all classes

        Target:
        calculate P(class|data) which means given a certain dataset, check what class the entry should be assigned to
        '''
        rows = len(self.data)
        self.calculate_p_cj(rows)
        for index, feature in enumerate(self.data):
            # calculate the P(cj)
            self.cj[self.target[index]] += 1
        self.cj = self.cj / len(self.target)
        for x in xi_cj:
            for key, value in self.xi_cj[x].items():
                self.xi_cj[x][key] = value / len(self.xi_cj[x])

    def predict(self, test_data):
        probability = 1
        result = {}
        for cj in self.xi_cj:
            for xi in test_data:
                result = result * cj[xi]
            result[cj] = result
            result = 1
        winner = [key for key, value in result.values() if value == max(result.values())]
        print(test_data, ' belongs to class ', winner)
        return winner, result[winner]

    def echo(self):
        print(self.data.head(10))
        print(self.cj)
        print(self.data.groupby(self.class_col)["P(xi|cj)"].sum())


if __name__ == '__main__':

    classifier = IrisNaiveBayesClassifier()
    classifier.calculate_p_c(len(classifier.data))
    classifier.calculate_p_xi_cj()
    classifier.echo()

