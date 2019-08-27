import numpy as np
import pandas as pd
from collections import namedtuple

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
    # from math import pi, exp
    # from numpy import exp, sqrt
    # from scipy.integrate import quad
    # return (1/sqrt(2*pi*std**2))*exp(-((x-mean)/std)**2/2)
    return norm.pdf(x, loc=mean, scale=std)


class IrisNaiveBayesClassifier:
    """
    Gaussian density distribution
    """

    def __init__(self, path="data/iris.data", class_col="class"):
        self.dtypes = {
            'sepal_length_in_cm': np.float
            , 'sepal_width_in_cm': np.float
            , 'petal_length_in_cm': np.float
            , 'petal_width_in_cm': np.float
        }
        self.feature_columns = list(self.dtypes.keys())
        columns = self.feature_columns + [class_col]
        self.data = pd.read_csv(path, dtype=self.dtypes)[columns]
        self.class_col = class_col
        self.classes = set(self.data[class_col])
        self.Parameter = namedtuple("Parameter", ["means", "stds", "probability"])
        self.parameters = {}

    def calculate_p_c(self):
        """
        P(c)
        """
        rows = len(self.data)
        self.data.groupby(self.class_col)
        self.data = pd.merge(self.data, (self.data[self.class_col].value_counts()/rows).to_frame().reset_index().rename(
                     columns={"index": "class", "class": "P(c)"}), on="class")

    def calculate_p_x(self, x):
        """
        This is called the normalizing constant, this is actually where the classified result change 
        p(x) = p(x1)*p(x2)*p(x3)...*p(xn)

        Since our goal is to get the corresponding class that has the highest possibilities and this term is
        not related to the corresponding class. Therefore we can ignore this term in practice
        """
        pass

    def calculate_p_x_c(self, df, cls, distribution_func=gaussian_distribution):
        """
        If use_bin, the continuous value will be distributed into different quartiles 

        In naive bayesian network, each feature is independent. Therefore, the P(data|class) could be separated

        This is called the hypothesis

        """
        result = 1
        for feature in self.dtypes:
            p_feature_class = distribution_func(df[feature], self.parameters[cls].means[feature], self.parameters[cls].stds[feature])
            # this is based on the assumption that each features are independent
            result *= p_feature_class
        return result

    def fit(self):
        """
        Bayesian thereom:
        P(data|class) * P(class) = P(class|data) * P(data)

        Assumption 1: xi are independent(otherwise, a bayesian network is used)
        P(data|cj) = P(x1|class_j) * P(x2|class_j) ... P(xn|class_j)

        P(class_j|data) = P(class_j) * P(data|class_j) / P(data)
        P(data) = SUM_j(P(data|class_j)*P(class_j))
        P(data) could be ignored since it is the same for all classes

        Target:
        fit the parameter of mean, std and p(c) for each class
        """
        rows = len(self.data)
        for clazz in self.classes:
            p_c = self.data[self.class_col].value_counts().loc[clazz]/rows
            mean = self.data.groupby(self.class_col).mean().loc[clazz]
            std = self.data.groupby(self.class_col).std().loc[clazz]
            self.parameters[clazz] = self.Parameter(mean, std, p_c)

    def predict(self, test_data):
        score = []
        test_data = test_data[self.feature_columns]
        for cls in self.parameters:
            temp_score = self.calculate_p_x_c(test_data, cls) * self.parameters[cls].probability
            score.append(temp_score)
        result_index = np.argmax(score, axis=1)
        result = [list(self.classes)[x] for x in result_index]
        test_data["predict_class"] = result
        print(test_data.head(10))
        return test_data

    def echo(self):
        print(self.data.head(10))
        print(self.parameters)


if __name__ == '__main__':
    classifier = IrisNaiveBayesClassifier()
    classifier.fit()
    test = pd.read_csv("data/iris.test")
    classifier.predict(test)

