import torch
import torch.nn as nn
import numpy as np

from utils.mri import coilcombine, ifft2c_mri
from .base import NIKBase

class NIKSiren(NIKBase):
    def __init__(self, config) -> None:
        super().__init__(config)
        B = torch.randn((self.config['coord_dim'], self.config['feature_dim']//2), dtype=torch.float32)
        self.register_buffer('B', B)
        self.create_network()
        self.to(self.device)
        
        
    def create_network(self):
        feature_dim = self.config["feature_dim"]
        num_layers = self.config["num_layers"]
        out_dim = self.config["out_dim"]
        self.network = Siren(feature_dim, num_layers, out_dim).to(self.device)

    def pre_process(self, inputs):
        """
        Preprocess the input coordinates.
        """
        inputs['coords'] = inputs['coords'].to(self.device)
        if inputs.keys().__contains__('targets'):
            inputs['targets'] = inputs['targets'].to(self.device)
        features = torch.cat([torch.sin(inputs['coords'] @ self.B),
                              torch.cos(inputs['coords'] @ self.B)] , dim=-1)
        inputs['features'] = features
        return inputs
    
    def post_process(self, output):
        """
        Convert the real output to a complex-valued output.
        The first half of the output is the real part, and the second half is the imaginary part.
        """
        output = torch.complex(output[...,0:self.config["out_dim"]], output[...,self.config["out_dim"]:])
        return output

    def train_batch(self, sample):
        self.optimizer.zero_grad()
        sample = self.pre_process(sample)
        output = self.forward(sample)
        output = self.post_process(output)
        loss, reg = self.criterion(output, sample['targets'], sample['coords'])
        loss.backward()
        self.optimizer.step()
        return loss
    
    def test_batch(self):
        """
        Test the network with a cartesian grid.
        if sample is not None, it will return image combined with coil sensitivity.
        """
        with torch.no_grad():
            nt = self.config['nt']
            nx = self.config['nx']
            ny = self.config['ny']

            ts = torch.linspace(-1+1/nt, 1-1/nt, nt)
            nc = len(self.config['coil_select'])
            kc = torch.linspace(-1, 1, nc)
            kxs = torch.linspace(-1, 1-2/nx, nx)
            kys = torch.linspace(-1, 1-2/ny, ny)
            

            # TODO: disgard the outside coordinates before prediction
            grid_coords = torch.stack(torch.meshgrid(ts, kc, kxs, kys, indexing='ij'), -1).to(self.device) # nt, nc, nx, ny, 4
            dist_to_center = torch.sqrt(grid_coords[:,:,:,:,2]**2 + grid_coords[:,:,:,:,3]**2)

            # split t for memory saving
            t_split = 1
            t_split_num = np.ceil(nt / t_split).astype(int)

            kpred_list = []
            for t_batch in range(t_split_num):
                grid_coords_batch = grid_coords[t_batch*t_split:(t_batch+1)*t_split]

                grid_coords_batch = grid_coords_batch.reshape(-1, 4).requires_grad_(False)
                # get prediction
                sample = {'coords': grid_coords_batch}
                sample = self.pre_process(sample)
                kpred = self.forward(sample)
                kpred = self.post_process(kpred)
                kpred_list.append(kpred)
            kpred = torch.concat(kpred_list, 0)
            
            # kpred_list.append(kpred)
            # kpred = torch.mean(torch.stack(kpred_list, 0), 0) #* filter_value.reshape(-1, 1)


            # TODO: clearning this part of code
            kpred = kpred.reshape(nt, nc, nx, ny)
            k_outer = 1
            kpred[dist_to_center>=k_outer] = 0
            # kpred = kpred.permute(0, 3, 1, 2)
            return kpred
    
    def forward(self, inputs):
        return self.network(inputs['features'])

"""
The following code is a demo of mlp with sine activation function.
We suggest to only use the mlp model class to do the very specific 
mlp task: takes a feature vector and outputs a vector. The encoding 
and post-process of the input coordinates and output should be done 
outside of the mlp model (e.g. in the prepocess and postprocess 
function in your NIK model class).
"""

class Siren(nn.Module):
    def __init__(self, hidden_features, num_layers, out_dim, omega_0=30, exp_out=True) -> None:
        super().__init__()

        self.net = [SineLayer(hidden_features, hidden_features, is_first=True, omega_0=omega_0)]
        for i in range(num_layers-1):
            self.net.append(SineLayer(hidden_features, hidden_features, is_first=False, omega_0=omega_0))
        final_linear = nn.Linear(hidden_features, out_dim*2)
        with torch.no_grad():
            final_linear.weight.uniform_(-np.sqrt(6 / hidden_features) / omega_0, 
                                          np.sqrt(6 / hidden_features) / omega_0)
        self.net.append(final_linear)
        self.net = nn.Sequential(*self.net)
    
    def forward(self, features):
        return self.net(features)



class SineLayer(nn.Module):
    """Linear layer with sine activation. Adapted from Siren repo"""
    def __init__(self, in_features, out_features, bias=True,
                 is_first=False, omega_0=30):
        super().__init__()
        self.omega_0 = omega_0
        self.is_first = is_first
        self.in_features = in_features
        self.linear = nn.Linear(in_features, out_features, bias=bias)

        with torch.no_grad():
            if self.is_first:
                self.linear.weight.uniform_(-1 / self.in_features, 
                                             1 / self.in_features)      
            else:
                self.linear.weight.uniform_(-np.sqrt(6 / self.in_features) / self.omega_0, 
                                             np.sqrt(6 / self.in_features) / self.omega_0)
        
    def forward(self, input):
        return torch.sin(self.omega_0 * self.linear(input))
    