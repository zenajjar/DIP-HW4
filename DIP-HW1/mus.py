import numpy as np
import pandas as pd

arr = np.array([1, 2, 3, 4, 5, 6, 6, 77, 77, 5, 5, 5, 5, 5])

value, count = np.unique(arr, return_counts=True)
temp = pd.DataFrame({"value": value, "counts": list(count)})

temp.counts = temp.counts / arr.shape[0]

print(temp.counts)

temp2 = np.array(temp.counts)

temp.counts = np.cumsum(temp2)

temp.counts = temp.counts * 255

temp.counts = temp.counts.astype(int)

print(temp)
