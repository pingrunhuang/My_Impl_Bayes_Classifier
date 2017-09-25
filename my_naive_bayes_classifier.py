import numpy as np

"""
Personal thought:
I was trying to implement the bayes classifier on a numerical dataset at the beginning.
But the problem with this is that the value of a numarical dataset is not that common to be the same (means 0.001 is not that common to be shown on feature 1)
So I think that is the reason why bayes is commonly used in NLP. 
"""

class My_Naive_Bayes_Classifier():
	self.score = null
	def __init__(self, data, target):
		self.data = data
		self.target = target
		# xi_cj denote { c1 :{x1:0,...xn:0}, c2:{x1:0,...xn:0} ... cm:{x1:0,...xn:0} }
		self.xi_cj = { cj :{} for cj in set(target) }
		self.cj = {x:0 for x in set(target)}

	def fit(self):
		'''
		P(D|cj) = P(x1|cj) * P(x2|cj) ... P(xn|cj)
		P(cj|D) = P(cj) * P(D|cj) / P(D)
		P(D) = SUM_j(P(D|C_j)*P(C_j))
		P(D) could be ignored since it is the same for all classes
		'''
		if len(self.data) != len(self.target):
			raise Execption("data and target is not compatible")

		for index, feature in enumerate(self.data):
			# calculate the P(cj)
			self.cj[self.target[index]] += 1

			for xi in feature:
				if xi not in self.xi_cj[self.target[index]]:
					self.xi_cj[self.target[index]][xi] = 1
				else:
					self.xi_cj[self.target[index]][xi] += 1

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

