##1
import numpy as np
def f(x):
  return 1/x
epsilon = np.finfo(np.float32).eps
for x in [1,-1]:
  slope=(f(x+epsilon)-f(x))/epsilon
  y = f(x)
  c = y - slope * x
  print("slope at x={} is {}".format(x,slope))
  print("tangent line is y={:f}x{:+f}".format(slope,c))
##2
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

##3
import numpy as np
def f(x):
  return x**2
epsilon = np.finfo(np.float32).eps
for x in [1,-1]:
  slope=(f(x+epsilon)-f(x))/epsilon
  y = f(x)
  c = y - slope * x
  print("slope at x={} is {}".format(x,slope))
  print("tangent line is y={:f}x{:+f}".format(slope,c))
  
##4
import numpy as np
import matplotlib.pyplot as plt

def f(x):
    return x**2

epsilon = np.finfo(np.float32).eps

# Values for x
x_values = np.linspace(-2, 2, 1000)

# Plot the function
plt.plot(x_values, f(x_values), label='f(x) = x^2')

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
plt.legend()

# Show the plot
plt.grid(True)
plt.show()

##5
import numpy as np
def f(x):
  return (x**3)+(2*x)+1
epsilon = np.finfo(np.float32).eps
for x in [1,-1]:
  slope=(f(x+epsilon)-f(x))/epsilon
  y = f(x)
  c = y - slope * x
  print("slope at x={} is {}".format(x,slope))
  print("tangent line is y={:f}x{:+f}".format(slope,c))
