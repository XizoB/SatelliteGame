import collections
import numpy as np


buffer = collections.deque(maxlen=3)
for i in range(8):
    buffer.append([i,i+1])

print(buffer)

a = np.array([1,2])
b = np.array([3,4])
print(a,b)
print(np.hstack([a,b]))