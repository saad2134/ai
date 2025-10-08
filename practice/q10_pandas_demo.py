import pandas as pd

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
