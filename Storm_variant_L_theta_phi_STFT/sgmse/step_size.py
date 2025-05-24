import numpy as np
import matplotlib.pyplot as plt

# Set the values for self.sde.T and self.t_eps
T = 1.0
t_eps = 0.0

# Number of steps you want to generate
num_steps = 30

# Generate uniform steps between 0 and 1
uniform_steps = np.linspace(0, 1, num_steps)

# Apply an inverse exponential transformation to create non-uniform steps
exponent = 4  # Adjust this exponent to control the growth rate
non_uniform_steps = (1 - uniform_steps)**exponent

# Normalize the non-uniform steps to ensure their sum is 1
non_uniform_steps -= non_uniform_steps.min()
non_uniform_steps /= non_uniform_steps.sum()

# Scale the non-uniform steps to the range [t_eps, T]
scaled_steps = non_uniform_steps * (T - t_eps) + t_eps

# Compute the step sizes (differences between consecutive steps)
step_sizes = np.diff(scaled_steps)

# Print the step sizes
print("Step Sizes:")
print(step_sizes)

# Plot the step sizes
plt.plot(np.arange(num_steps - 1), step_sizes, marker='o', linestyle='-', markersize=5)
plt.title('Non-Uniform Step Sizes (Larger towards 0, Smaller towards 1)')
plt.xlabel('Step Index')
plt.ylabel('Step Size')
plt.grid(True)
plt.show()

