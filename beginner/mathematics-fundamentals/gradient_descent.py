def gradient_descent(f, df, start, learning_rate=0.1, n_iter=100):
    x = start
    for i in range(n_iter):
        grad = df(x)
        x = x - learning_rate * grad
        if i % 20 == 0:
            print(f"Iteration {i}: x = {x:.4f}, f(x) = {f(x):.4f}")
    return x

def f(x):
    return x**2

def df(x):
    return 2*x

minimum = gradient_descent(f, df, start=5, learning_rate=0.1)
print(f"Found minimum at x = {minimum:.4f}")