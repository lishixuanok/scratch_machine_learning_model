import math
import numpy as np
import pandas as pd
from functools import reduce


class decision_tree():
    def __init__(self, df):
        self.df = df

    def max_information_gain_attribute(self, attribute, value):
        columns = self.df.columns.values
        gain_result = {}
        for column in columns:
            if column == attribute:
                continue
            column_values = df[column].unique()
            ent, probability = [], []
            for _value in column_values:
                d_probability = len(self.df.loc[(self.df[column] == _value) & (self.df[attribute] == value)]) / \
                                len(self.df[self.df[column] == _value])
                ent.append(-(d_probability * math.log(d_probability, 2) +
                             (1 - d_probability) * math.log((1 - d_probability), 2)))
                probability.append(len(self.df.loc[(self.df[column] == _value)]) / len(self.df.index))
            probability_total = len(self.df.loc[self.df[attribute] == value]) / len(self.df.index)
            ent_total = -(probability_total * math.log(probability_total, 2) +
                          (1 - probability_total) * math.log(1 - probability_total, 2))
            gain = ent_total - reduce(lambda x, y: x + y, [a * b for a, b in zip(ent, probability)])
            gain_result.update({column: gain})
        return max(gain_result, key=gain_result.get)


if __name__ == "__main__":
    df = pd.DataFrame(np.random.randint(low=0, high=2, size=(100000, 5)),
                      columns=['a', 'b', 'c', 'd', 'e'])
    calc = decision_tree(df)
    result = calc.max_information_gain_attribute('e', 1)
    print(result)
