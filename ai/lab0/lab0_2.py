from collections import Counter
import re
import pandas as pd
import matplotlib.pyplot as plt

df = pd.read_csv('winemag-data-130k-v2.csv')

words = Counter()
for review in df['description']:
    words.update(re.split('\W+', review.lower()))
words.pop('')

index, data = zip(*words.most_common(20))
pd.Series(data, index).plot.bar(grid=True)

plt.show()