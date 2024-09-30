import torch
import torch.nn as nn

class GAN(nn.Module):
    def __init__(self, generator, descriminator, criterion, noise_dim):
        super(GAN, self).__init__()
        self.generator = generator
        self.descriminator = descriminator
        self.criterion = criterion
        self.noise_dim = noise_dim

    def __getSeed(self, n_seeds):
        return torch.randn(n_seeds, self.noise_dim)

    def train(self, data, optimizer_g, optimizer_d, batch_size: int, n_iter: int, n_discrimitor_inner_step: int, n_generator_inner_step: int):
        losses_d, losses_g = [], []
        for _ in range(n_iter):

            for _ in range(n_discrimitor_inner_step):
                optimizer_d.zero_grad()

                x_data = data[torch.randperm(data.shape[0])[:batch_size]]
                x_gen = self.generator(self.__getSeed(n_seeds=batch_size))

                loss_data = self.criterion(self.descriminator(x_data), torch.ones(batch_size,1).to(x_data.device))
                loss_gen  = self.criterion(self.descriminator(x_gen), torch.zeros(batch_size,1).to(x_data.device))
                loss_d = loss_data + loss_gen
                loss_d.backward()
                optimizer_d.step()
            losses_d.append(loss_d)

            for _ in range(n_generator_inner_step):
                optimizer_g.zero_grad()

                x_gen = self.generator(self.__getSeed(n_seeds=batch_size))
                loss_g = -self.criterion(self.descriminator(x_gen), torch.zeros(batch_size,1).to(x_data.device))
                loss_g.backward()
                optimizer_g.step()
            losses_g.append(losses_g)

        return losses_d, losses_g

    def sample(self, n_points:int=1):
        seeds = self.__getSeed(n_seeds=n_points)
        return self.generator(seeds).detach()
    
if __name__ == "__main__":
    import torch
    import torch.optim as optim
    import matplotlib.pyplot as plt

    import sys
    sys.path.append("/Users/tiejean/Documents/Education/Implementations/mlHlpers")

    from modules.mlp import MLP
    from modules.gan import GAN
    from utils.data import sample_circular_gmm

    data, data_info = sample_circular_gmm(5000, num_modes=4, viz=True)

    batch_size = 1000
    data_dim = 2
    noise_dim = 2
    generator = MLP([noise_dim, 128, 128, 128, 128, 128, data_dim])
    discriminator = MLP([data_dim, 128, 128, 128, 1])
    gan = GAN(generator, discriminator, nn.BCEWithLogitsLoss(), noise_dim=noise_dim)

    optimizer_g = optim.Adam(generator.parameters(), lr=0.0005, betas = (0.5, 0.999))
    optimizer_d = optim.Adam(discriminator.parameters(), lr=0.0005, betas= (0.5, 0.999))

    gan.train(data, optimizer_g, optimizer_d, batch_size, n_iter=1000, n_discrimitor_inner_step=10, n_generator_inner_step=1)