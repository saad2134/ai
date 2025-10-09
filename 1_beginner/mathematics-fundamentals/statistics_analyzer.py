import statistics

def analyze_data(data):
    return {
        'mean': statistics.mean(data),
        'median': statistics.median(data),
        'stdev': statistics.stdev(data),
        'variance': statistics.variance(data)
    }

data = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
stats = analyze_data(data)
print(f"Mean: {stats['mean']}, Median: {stats['median']}, Std: {stats['stdev']:.2f}")