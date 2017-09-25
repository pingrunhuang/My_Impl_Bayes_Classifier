import numpy as np
from sklearn import datasets

data = datasets.load_iris()

class My_Naive_Bayes_Classifier():
	self.score = null
	def __init__(self, data, target):
		self.data = data
		self.target = target

	def fit(self):
		'''
		P(D|cj) = P(x1|cj) * P(x2|cj) ... P(xn|cj)
		P(cj|D) = P(cj) * P(D|cj) / P(D)
		P(D) = SUM_j(P(D|C_j)*P(C_j))
		P(D) could be ignored since it is the same for all classes
		'''
		if len(self.data) != len(self.target):
			raise Execption("data and target is not compatible")

		self.cj={}
		self.xi_cj={}
		for index, feature in enumberate(self.data):
			# calculate the P(cj)
			if self.target[index] not in self.cj:
				self.cj[self.target[index]] = 1
				# initialize the dictionary of conditional possibilities
				self.xi_cj[self.target[index]] = {}
			else:
				self.cj[self.target[index]] += 1

			for xi in feature:
				if xi not in self.xi_cj[self.target[index]]:
					self.xi_cj[self.target[index]][xi] = 1
				else:
					self.xi_cj[self.target[index]][xi] += 1

		self.cj = self.cj / len(self.target)
		for x in xi_cj:
			self.xi_cj[x] = self.xi_cj[x] / len(self.xi_cj[x])

	def predict(self, test_data):
		pass

