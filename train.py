import torch
import argparse
import numpy as np
import random

import wandb
from datasets.cardiac import RadialDataset
from models.siren import NIKSiren
# from models.insngp_tcnn import NIKHashSiren
from utils.basic import parse_config
from torch.utils.data import DataLoader

from utils.vis import angle2color, k2img

def main():
    # parse args and get config
    parser = argparse.ArgumentParser()
    parser.add_argument('-c', '--config', type=str, default='configs/config.yml')
    parser.add_argument('-g', '--gpu', type=int, default=0)
    parser.add_argument('-s', '--slice_name', type=str, default='CINE_S1_rad_AA')
    parser.add_argument('-seed', '--seed', type=int, default=0)
    # parser.add_argument('-s', '--seed', type=int, default=0)
    args = parser.parse_args()

    # enable Double precision
    torch.set_default_dtype(torch.float32)

    # set gpu and random seed
    torch.cuda.set_device(args.gpu)
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.random.manual_seed(args.seed)

    # parse config
    slice_name = args.slice_name
    config = parse_config(args.config)
    config['slice_name'] = slice_name
    config['gpu'] = args.gpu

    # create dataset
    dataset = RadialDataset(config)
    dataloader = DataLoader(dataset, batch_size=config['batch_size'], shuffle=True, num_workers=config['num_workers'])
    # config['eps'] = dataset.eps
    # create model
    if config['model'] == 'siren':
        NIKmodel = NIKSiren(config)
    # elif config['model'] == 'hashsiren':
    #     NIKmodel = NIKHashSiren(config)

    NIKmodel.init_train()

    for epoch in range(config['num_steps']):
        loss_epoch = 0
        for i, sample in enumerate(dataloader):
            # kcoord, kv = sample['coords'], sample['target']
            loss = NIKmodel.train_batch(sample)
            print(f"Epoch: {epoch}, Iter: {i}, Loss: {loss}")
            loss_epoch += loss
        

        kpred = NIKmodel.test_batch()

        # kpred[kpred != 0] = (dataset.eps / torch.abs(kpred[kpred != 0]) - dataset.eps) * (kpred[kpred != 0] / torch.abs(kpred[kpred != 0]))

        vis_img = k2img(kpred, dataset.csm)
        
        log_dict = {
            'loss': loss_epoch/len(dataloader),
            'k': wandb.Video(vis_img['k_mag'], fps=10, format="gif"),
            'img': wandb.Video(vis_img['combined_mag'], fps=10, format="gif"),
            'img_phase': wandb.Video(vis_img['combined_phase'], fps=10, format="gif"),
            'khist': wandb.Histogram(torch.view_as_real(kpred).detach().cpu().numpy().flatten()),
        }
        NIKmodel.exp_summary_log(log_dict)



if __name__ == '__main__':
    main()