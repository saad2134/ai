import numpy as np
import pandas as pd
from scipy import linalg, optimize



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
print("5 numbers between 0 and 1:\n",a3)

# Method 8 dot() - Dot product of 2 arrays
print("Dot product of values of a:\n",np.dot(a,a))

# Method 9 median() - median of values of an array
print("Median of values of a:\n",np.median(a))

# Method 10 std() - standard deviation of values of an array
print("Standard deviation of values of a:\n",np.std(a))



print('---------------------------------------------------------------------------------------------------------')



#A: Creation of Dataframe from dictionary
data = {
    "Name": ["Alice", "Bob", "Charlie"],
    "Age": [20,24,28],
    "City": ["New York", "London", "Paris"]
}
df1 = pd.DataFrame(data)
print(df1,"\n")

#B: Display the top 10 and bottom 10 rows from the dataframe
df2=pd.DataFrame({'num':range(1,21)})
print("Top 10:\n",df2.head(10))
print("Bottom 10:\n",df2.tail(10))

#C: Display the dimensions of the dataframe
df3=pd.DataFrame({'x':[1,2,3],'y':[4,5,6]})
print("\nDimensions:",df3.shape)

#D: Display the row at index 3
df4=pd.DataFrame({'x':[10,20,30,40],'y':[100,200,300,400]})
print("\nRow at index 3:\n",df4.loc[3])



print('-----------------------------------------------------------------------------------------------------------')



# Solving Linear Equations
# 2x + 3y = 8
# 5x + 4y = 13
A = np.array([[2, 3],
              [5, 4]])
B = np.array([8, 13])
solution = linalg.solve(A, B)
print("x =", solution[0])
print("y =", solution[1])


#Find the Minimum of a Function (Optimization)
# Function: f(x) = x^2 + 4x + 4
def f(x):
    return x**2 + 4*x + 4
result = optimize.minimize(f, x0=0)
print("Minimum value:", result.fun)
print("At x =", result.x)



