class MMProblem:
    def __init__(self):
        self.mixture_color_1994 = {
            "brown": 0.3,
            "yellow": 0.2,
            "red": 0.2,
            "green": 0.1,
            "orange": 0.1,
            "tan": 0.1
        }
        self.mixture_color_1996 = {
            "blue": 0.24,
            "green": 0.2,
            "orange": 0.16,
            "yellow": 0.14,
            "red": 0.13,
            "brown": 0.13
        }
        self.colors = set([x for x in self.mixture_color_1994] + [x for x in self.mixture_color_1996])

    def calculate(self, mm_color1="yellow", mm_color2="green"):
        """
        tell the probabilities that bag1 is from 1994 and the probabilities that bag1 is from 1996
        :param mm_color1: the color of the mm bean drawn from bag1
        :param mm_color2: the color of the mm bean drawn from bag2
        :return: the probability of bag1 is from 1994 and the probability of bag1 is from 1996 respectively
        """
        if mm_color1 not in self.colors or mm_color2 not in self.colors:
            raise ValueError("No such color in bags")
        # the probability that bean1 is from 1994
        prior1 = 0.5
        # the probability that bean1 is from 1996
        prior2 = 0.5
        # bean1 with mm_color1 is from 1994 and bean2 with mm_color2 is from 1996
        likelihood1 = self.mixture_color_1994[mm_color1] * self.mixture_color_1996[mm_color2] * prior1
        # bean1 with mm_color1 is from 1996 and bean2 with mm_color2 is from 1994
        likelihood2 = self.mixture_color_1996[mm_color1] * self.mixture_color_1994[mm_color2] * prior2
        normalizing_constant = likelihood1+likelihood2
        posterior1 = likelihood1/normalizing_constant
        posterior2 = likelihood2/normalizing_constant
        return posterior1, posterior2


if __name__ == '__main__':
    solver = MMProblem()
    print(solver.calculate())
