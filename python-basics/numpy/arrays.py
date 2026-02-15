import numpy as np

# simple array
a = np.array([1, 2, 3, 4])
print(a)


# 2d array
b = np.array([[1, 2], [3, 4]])
print(b)


# mean and sum
a = np.array([10, 20, 30, 40])

print(np.mean(a))
print(np.sum(a))


# reshape
a = np.array([1, 2, 3, 4, 5, 6])
b = a.reshape(2, 3)

print(b)
