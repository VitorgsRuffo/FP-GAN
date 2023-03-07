timestamp = [i for i in range(0,86400)]
print(timestamp)
import numpy as np

timestamp = np.array(timestamp)
timestamp = timestamp[(0, -3:-1)]



# unique, counts = np.unique(res, return_counts=True)
# print(dict(zip(unique, counts)))

# unique, counts = np.unique(labels, return_counts=True)
# print(dict(zip(unique, counts)))

# input()