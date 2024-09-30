import torch
import torch.nn as nn
import math

class EnergyModel(nn.Module):
    def __init__(self, model, input_dim):
        super(EnergyModel, self).__init__()
        self.energy = model  #  the negative energy
        self.mean_prior = nn.Parameter(torch.zeros(input_dim))
        self.inv_std_prior = nn.Parameter(torch.ones(input_dim))
        self.temperature = 1.0

    # Add a guassian prior to regularize the model,
    # guiding it to prefer state x that are more probable under a Gaussian distribution
    def gaussian_prior(self, x):
        return torch.sum(0.5 * self.inv_std_prior * (x - self.mean_prior)**2, dim=-1, keepdim=True)

    def forward(self, x):
        # low-energy states correspond to high-probability outcomes
        return - (self.energy(x) + self.gaussian_prior(x)) / self.temperature 
    
    def langevin_sampler(self, x0, n_steps=100, step_size=0.01, record_traj = False):
        x = x0.detach().requires_grad_(True)  # Ensure x requires grad for sampling
        if record_traj: traj = [x.detach()]
        for _ in range(n_steps):
            grad_energy = torch.autograd.grad(self(x).sum(), x)[0]
            x = x + .5*step_size * grad_energy + torch.randn_like(x) * math.sqrt(2 * step_size)
            if record_traj: traj.append(x.detach())
        if record_traj:
          return x.detach(), torch.stack(traj)
        else:
          return x.detach()

    def compute_loss(self, x, n_steps=1, step_size=1e-3):
        # The idea behind using data_real as the starting point for the Langevin sampler is that 
        # real data already represents a reasonable approximation of high-probability regions of the model. 
        # By perturbing real data only slightly, we can generate negative samples that are near the real data 
        # and fall into lower-probability areas under the current model.
        xprime = self.langevin_sampler(x, n_steps=n_steps, step_size=step_size).detach()
        # The loss here is not trying to directly measure the distance between energies 
        # or to penalize deviations in a squared sense, like in regression. 
        # Instead, it’s designed to compare the relative energy levels between real and fake data.
        return - (self.energy(x).mean() - self.energy(xprime).mean())
    
    def train(self, data, optimizer, batch_size: int, n_iter: int):
        for _ in range(n_iter):
            optimizer.zero_grad()
            x_data = data[torch.randperm(len(data))[:batch_size]]
            loss = self.compute_loss(x_data)
            loss.backward()
            optimizer.step()