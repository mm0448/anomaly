import torch
import torch.nn as nn


class SimpleVAE(nn.Module):
    def __init__(
        self, 
        dim_x: int,
        dim_h: int,
        dim_z: int
    ) -> None:
        """
        Basic baseline VAE to train from scratch
        """
        super().__init__()
        self.dim_x = dim_x
        self.dim_h = dim_h
        self.dim_z = dim_z

        self.hidden_layer_enc = torch.nn.Sequential(
            nn.Linear(self.dim_x, self.dim_h),
            nn.ReLU()
        )
        self.mean_layer = nn.Linear(self.dim_h, self.dim_z)
        self.logvar_layer = nn.Linear(self.dim_h, self.dim_z)
        self.decoder = torch.nn.Sequential(
            nn.Linear(self.dim_z, self.dim_h),
            nn.ReLU(),
            nn.Linear(self.dim_h, self.dim_x)
        )

    def sample(self, z_mean: torch.Tensor, z_logvar: torch.Tensor):
        z_std = torch.exp(0.5 * z_logvar)
        eps = torch.randn_like(z_std, requires_grad=False).detach()
        return z_mean + eps * z_std
    
    def encode(self, x: torch.Tensor):
        h = self.hidden_layer_enc(x)
        z_mean = self.mean_layer(h)
        z_logvar = self.logvar_layer(h)
        z = self.sample(z_mean, z_logvar)
        return z, z_mean, z_logvar
    
    def decode(self, z: torch.Tensor):
        return self.decoder(z)
    
    def forward(self, x: torch.Tensor):
        z, z_mean, z_logvar = self.encode(x)
        x_reconst = self.decode(z)
        return x_reconst, z_mean, z_logvar
