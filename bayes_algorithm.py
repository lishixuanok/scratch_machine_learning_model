import pandas as pd
import numpy as np
from functools import reduce


class bayes():
    """
    p(c|x) = p(x|c)*p(c)/p(x)
    priori_probability: p(c), p(x)
    conditional_probability: p(x|c)
    """

    def __init__(self, df, sample):
        self.df = df
        self.sample = sample

    def priori_probability(self, attribute, value):
        total = len(self.df.index)
        total_attribute = len(self.df[self.df[attribute] == value])
        return total_attribute / total

    def conditional_probability(self, attribute, value):
        c_total = len(self.df[self.df[attribute] == value])
        conditional_result = [len(self.df.loc[(self.df[key_] == value_) & (self.df[attribute] == value)]) / c_total
                              for key_, value_ in self.sample.items()]
        return reduce(lambda x, y: x * y, conditional_result)


def run():
    df = pd.DataFrame(np.random.randint(low=0, high=2, size=(100000, 5)),
                      columns=['a', 'b', 'c', 'd', 'e'])
    calc_b = bayes(df,
                   sample={'a': 1, 'b': 0, 'c': 0, 'd': 1})

    # e:0
    prior = calc_b.priori_probability('e', 0)
    conditional = calc_b.conditional_probability('e', 0)
    result = conditional * prior
    print('e:0', result)

    # e:1
    prior = calc_b.priori_probability('e', 1)
    conditional = calc_b.conditional_probability('e', 1)
    result = conditional * prior
    print('e:1', result)


if __name__ == '__main__':
    run()
