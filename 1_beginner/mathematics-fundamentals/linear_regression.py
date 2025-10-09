def linear_regression(x, y):
    n = len(x)
    x_mean = sum(x) / n
    y_mean = sum(y) / n
    
    numerator = sum((x[i] - x_mean) * (y[i] - y_mean) for i in range(n))
    denominator = sum((x[i] - x_mean) ** 2 for i in range(n))
    
    m = numerator / denominator
    b = y_mean - m * x_mean
    return m, b

x = [1, 2, 3, 4, 5]
y = [2, 4, 5, 4, 5]
m, b = linear_regression(x, y)
print(f"y = {m:.2f}x + {b:.2f}")