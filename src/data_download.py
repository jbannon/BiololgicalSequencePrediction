import sys
import pandas as pd
from tdc.single_pred import Develop



data = Develop(path = "../data/",name = 'SAbDab_Chen')


dataset = data.get_data()

print(dataset)
print(dataset.columns)

