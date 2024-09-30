import torch
import torch.nn as nn

class RectifiedFlow(nn.Module):
    def __init__(self, model, dim):
        super(RectifiedFlow, self).__init__()
        self.model = model 
        self.dim = dim

    def forward(self, x, t):
        x = x.view(x.shape[0], -1)
        t = t.view(t.shape[0], -1)
        xt = torch.cat((x, t), dim=1).float()
        return self.model(xt)
    
    def train(self, data, optimizer, batch_size: int, n_iter: int):
        losses = []
        for _ in range(n_iter):
            optimizer.zero_grad()
            X1 = data[torch.randperm(len(data))][:batch_size]
            X0 = torch.randn_like(X1)
            t = torch.rand((batch_size, 1)).to(X1.device)
            # get training pairs
            Xt = t * X1 + (1-t) * X0
            dot_Xt = X1 - X0
            # MSE loss
            loss = torch.mean((self(Xt, t) - dot_Xt)**2)
            loss.backward()
            optimizer.step()
            losses.append(loss.item())
        return losses
    
    def euler_sample(self, n_points:int=1, n_steps:int=100):
        xt = torch.randn((n_points, self.dim))
        
        with torch.no_grad():
            for step in range(n_steps):
                t = (step / n_steps); step_size = 1/n_steps
                t_ones = t*torch.ones(xt.shape[0], 1)
                v_t = self(xt, t_ones.squeeze())
                xt = xt + step_size *  v_t
        return xt
    
if __name__ == "__main__":
    import torch
    import torch.optim as optim
    import matplotlib.pyplot as plt

    import sys
    sys.path.append("/Users/tiejean/Documents/Education/Implementations/mlHlpers")

    from modules.mlp import MLP
    from modules.rectified_flow import RectifiedFlow
    from utils.data import sample_circular_gmm

    data, data_info = sample_circular_gmm(5000, num_modes=4, viz=True)

    batch_size = 1000
    dim = 2
    mlp = MLP([dim+1, 64, dim])
    rectified_flow = RectifiedFlow(mlp, dim)

    optimizer = optim.Adam(rectified_flow.parameters(), lr=5e-4)
    losses = rectified_flow.train(data, optimizer, batch_size, n_iter=1000)