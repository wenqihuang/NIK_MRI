import torch

class HDRLoss_FF(torch.nn.Module):
    """
    HDR loss function with frequency filtering (v4)
    """
    def __init__(self, config):
        super().__init__()
        self.sigma = float(config['hdr_ff_sigma'])
        self.eps = float(config['hdr_eps'])
        self.factor = float(config['hdr_ff_factor'])

    def forward(self, input, target, kcoords, weights=None, reduce=True):
        # target_max = target.abs().max()
        # target /= target_max
        # input = input / target_max
        # input_nograd = input.clone()
        # input_nograd = input_nograd.detach()
        dist_to_center2 = kcoords[...,1]**2 + kcoords[...,2]**2
        filter_value = torch.exp(-dist_to_center2/(2*self.sigma**2)).unsqueeze(-1)

        if input.dtype == torch.float:
            input = torch.view_as_complex(input) #* filter_value
        if target.dtype == torch.float:
            target = torch.view_as_complex(target)
        
        assert input.shape == target.shape
        error = input - target
        # error = error * filter_value

        loss = (error.abs()/(input.detach().abs()+self.eps))**2
        if weights is not None:
            loss = loss * weights.unsqueeze(-1)

        reg_error = (input - input * filter_value)
        reg = self.factor * (reg_error.abs()/(input.detach().abs()+self.eps))**2
        # reg = torch.matmul(torch.conj(reg).t(), reg)
        # reg = reg.abs() * self.factor
        # reg = torch.zeros([1]).mean()

        if reduce:
            return loss.mean() + reg.mean(), reg.mean()
        else:
            return loss, reg
        

class AdaptiveHDRLoss(torch.nn.Module):
    """
    HDR loss function with frequency filtering (v4)
    """
    def __init__(self, config):
        super().__init__()
        self.sigma = float(config['hdr_ff_sigma'])
        self.eps = float(config['eps'])
        self.factor = float(config['hdr_ff_factor'])

    def forward(self, input, target, reduce=True):
        # target_max = target.abs().max()
        # target /= target_max
        # input = input / target_max
        # input_nograd = input.clone()
        # input_nograd = input_nograd.detach()

        if input.dtype == torch.float:
            input = torch.view_as_complex(input) #* filter_value
        if target.dtype == torch.float:
            target = torch.view_as_complex(target)
        
        assert input.shape == target.shape
        error = input - target
        # error = error * filter_value

        loss = (-error.abs()/((input.detach().abs()+self.eps)**2))**2
        # if weights is not None:
        #     loss = loss * weights.unsqueeze(-1)

        # reg_error = (input - input * filter_value)
        # reg = self.factor * (reg_error.abs()/(input.detach().abs()+self.eps))**2
        # reg = torch.matmul(torch.conj(reg).t(), reg)
        # reg = reg.abs() * self.factor
        # reg = torch.zeros([1]).mean()

        if reduce:
            return loss.mean()