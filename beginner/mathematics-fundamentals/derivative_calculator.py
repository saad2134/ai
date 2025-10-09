def derivative(f, x, h=1e-5):
    return (f(x + h) - f(x)) / h

def f(x):
    return 3*x**2 + 2*x + 1

x = 2
df_dx = derivative(f, x)
print(f"f'({x}) = {df_dx:.2f}")