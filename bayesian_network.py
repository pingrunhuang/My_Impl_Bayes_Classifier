import pandas as pd
from abc import ABC

"""
Essentially, the bayesian network should be a DAG which describe the dependencies between each feature(aka column in dataset). This is also the fundamental difference between naive bayesian classifier and bayesian network classifier.

https://www.ibm.com/support/knowledgecenter/en/SS3RA7_15.0.0/com.ibm.spss.modeler.help/bayesian_networks_node_general.htm
https://www.bayesserver.com/examples/networks/asia#
"""


class BayesianNetwork(ABC):
    
    def fit(self):
        raise NotImplementedError("Please implement the fit method first")

    def classify(self):
        raise NotImplementedError("Please implement the classify method first")


class AsiaBayesianNetwork(BayesianNetwork):

    def __init__(self, path="data/ASIA10k.csv"):
        features = ["has_visited_asia", "has_tuberculosis", "smoking", "has_lung_cancer", "has_bronchitis", "either_tuberculosis_or_cancer", "xray_result", "dyspnea"]
        self.data = pd.read_csv("data/ASIA10k.csv", names=features, header=1)
        for col in features:
            self.data[col] = self.data[col].apply(lambda cell: True if cell=='yes' else False)
            self.data[col].astype("bool", inplace=True)
        
    def fit(self):
        pass
    
    def classify(self, test_data):

        pass

    def make_dependencies(self, is_draw=True):
        """
        Let's make the follow assumption:
        has_visited_asia -> has_tuberculosis -> either_tuberculosis_or_cancer -> dyspnea
        has_visited_asia -> has_tuberculosis -> either_tuberculosis_or_cancer -> xray_result
        smoking -> has_lung_cancer -> either_tuberculosis_or_cancer -> dyspnea
        smoking -> has_bronchitis -> dyspnea
        """

    def echo(self):
        print(self.data.head(10))

if __name__ == '__main__':

    asia = AsiaBayesianNetwork()
    asia.echo()


    