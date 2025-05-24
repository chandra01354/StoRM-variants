import torch
import matplotlib.pyplot as plt

class SDE:
    def __init__(self, T, t_eps):
        self.T = T
        self.t_eps = t_eps

class TimestepsGenerator:
    def __init__(self, sde):
        self.sde = sde

    def non_uniform_timesteps(self, x, lambda_=2):
        # Step 1: Generate uniform samples in [0, 1]
        uniform_samples = torch.rand(x.shape[0], device=x.device)

        # Step 2: Apply exponential transformation
        transformed_samples = -torch.log(1 - uniform_samples) / lambda_

        # Normalize the samples to [0, 1] range
        normalized_samples = transformed_samples / (transformed_samples.max() + 1e-6)

        # Step 3: Scale to [t_eps, T]
        t = self.sde.t_eps + (self.sde.T - self.sde.t_eps) * normalized_samples
        print(t)
        return t

# Parameters
T = 1.0
t_eps = 0.0
num_samples = 8  # Number of samples to generate
lambda_ = 2  # Rate parameter for the exponential distribution

# Initialize the SDE and generator
sde = SDE(T, t_eps)
generator = TimestepsGenerator(sde)

# Generate timesteps
timesteps = generator.non_uniform_timesteps(torch.empty(num_samples), lambda_=lambda_).cpu().numpy()

# Plot the distribution of timesteps
plt.hist(timesteps, bins=50, density=True, alpha=0.6, color='b', edgecolor='black')
plt.title('Distribution of Exponentially-Transformed Timesteps')
plt.xlabel('Timesteps')
plt.ylabel('Density')
plt.grid(True)
plt.show()

