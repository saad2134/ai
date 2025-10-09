import math

def covariance_correlation(x, y):
    n = len(x)
    x_mean = sum(x) / n
    y_mean = sum(y) / n
    
    cov = sum((x[i] - x_mean) * (y[i] - y_mean) for i in range(n)) / n
    corr = cov / (math.sqrt(sum((xi - x_mean)**2 for xi in x) / n) * 
                  math.sqrt(sum((yi - y_mean)**2 for yi in y) / n))
    return cov, corr

x = [1, 2, 3, 4, 5]
y = [2, 4, 6, 8, 10]
cov, corr = covariance_correlation(x, y)
print(f"Covariance: {cov:.2f}, Correlation: {corr:.2f}")