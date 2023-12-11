import numpy as np
import matplotlib.pyplot as plt

def f(x):
    return 1 / x

epsilon = np.finfo(np.float32).eps

# Values for x
x_values = np.linspace(-5, 5, 1000)

# Plot the function
plt.plot(x_values, f(x_values), label='f(x) = 1/x')

# Calculate and plot tangent lines
for x in [1, -1]:
    slope = (f(x + epsilon) - f(x)) / epsilon
    y_intercept = f(x) - slope * x
    tangent_line = slope * x_values + y_intercept
    plt.plot(x_values, tangent_line, label=f'Tangent at x={x}')


# Add labels and legend
plt.xlabel('x')
plt.ylabel('y')
plt.title('Function and Tangent Lines')
plt.grid(True)
plt.legend()

# Show the plot
plt.show()
