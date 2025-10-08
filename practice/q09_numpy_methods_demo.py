import numpy as np

# Method 1 array() - defining an  array
a=np.array([1,2,3,4])
print("a:",a)

# Method 2 sum() - sum of values of an array
print("Sum:",a.sum())

# Method 3 prod() - product of values of an array
print("Product:",a.prod())

# Method 4 mean() - mean of an array
print("Mean:",a.mean())

# Method 5 reshape() - changes the shape of an array
print("Reshape 2x2:\n",a.reshape(2,2))

# Method 6 arange() - creates values within a given interval
a2 = np.arange(0, 10, 2)
print("Range of 0 and 10 at 2:\n",a2)

# Method 7 linspace() - specific no. of numbers between 2 values
a3 = np.linspace(0, 1, 5)
print("5 numbers between 0 and 1:\n",a2)

# Method 8 dot() - Dot product of 2 arrays
print("Dot product of values of a:\n",np.dot(a,a))

# Method 9 median() - median of values of an array
print("Median of values of a:\n",np.median(a))

# Method 10 std() - standard deviation of values of an array
print("Standard deviation of values of a:\n",np.std(a))


