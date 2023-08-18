import os
from torch.utils.data import Dataset
import torch
import numpy as np
import h5py
import medutils


class RadialDataset(Dataset):
    def __init__(self, config):
        super().__init__()
        self.config = config
        
        file_folder = f"{config['data_root']}/{config['slice_name']}"
        self.data = self.get_subject_data(file_folder, config['num_cardiac_cycles'])
        # kdata: nc, nSpokes, nFE
        # traj: 2, nSpokes, nFE
        self.kdata_flat = np.reshape(self.data['kdata'].astype(np.complex64), (-1, 1))     # nc*nSpokes*nFE, 1

        self.kdata_org = self.data['kdata'].astype(np.complex64)
        nc, nSpokes, nFE = self.data['kdata'].shape
        self.n_kpoints = self.kdata_flat.shape[0]
        
        # create coordinates from trajectory
        traj = self.data['ktraj']    # has been normalized to [-1, 1] # 2, nSpokes, nFE
        tspokes = self.data['tspokes']  # nSpokes


        kcoords = np.zeros((nc, nSpokes, nFE, 4))
        kcoords[:,:,:,0] = np.reshape(tspokes * 2 - 1, (1, nSpokes, 1)) # normalize to [-1, 1]
        kc = torch.linspace(-1, 1, nc)
        kcoords[:,:,:,1] = np.reshape(kc, [nc, 1, 1])                   # nc, 1, 1
        kcoords[:,:,:,2] = traj[0, :][None]                             # 1, nSpokes, nFE
        kcoords[:,:,:,3] = traj[1, :][None]                             # 1, nSpokes, nFE


        self.kcoords_flat = np.reshape(kcoords.astype(np.float32), (-1, 4))

        # ############################
        # # eps = np.median(np.abs(self.kdata_flat))
        # eps = np.mean(np.abs(self.kdata_flat)) * 2
        # # # self.kdata_flat = eps / (self.kdata_flat + eps)
        # self.kdata_flat = eps / (np.abs(self.kdata_flat) + eps) * (self.kdata_flat / np.abs(self.kdata_flat))
        # self.eps = eps
        # ############################

        # self.device = torch.device('cuda')
        self.kdata_flat = torch.from_numpy(self.kdata_flat)#.to(device)     # nc*nSpokes*nFE
        self.kcoords_flat = torch.from_numpy(self.kcoords_flat)#.to(device) # nc*nSpokes*nFE, 4
        

    def align_to_one_cycle(self, ecg, acq_t0, TR, n_spokes, num_cycles):
        r"""Retrospective binning of cardiac data to a target number of cardiac cycles.
        Args:
            ecg: array with the timings of the ECG R-wave detections
            acq_t0: start time of acquisitions (same time axis as ecg)
            TR: repetition time
            n_spokes: number of shots
            num_cycles: number of cardiac cycles used for alignment

        Returns:
            data_binning(dict):
                'target_RR': target RR interval
                'num_cycles': number of cardiac cycles
                'tspokes': sorted relative time stamps of the spokes
                'idxspokes': indices of the spokes in the original data (sorted by the relative time in target_RR)
        """

        # discard some shots at the beginning of the data to avoid the 
        # influcence of transient magnetization
        # Ndiscard = np.ceil(self.config['transient_magnetization'] / TR)
        while ecg[0] < self.config['transient_magnetization']:
            ecg = ecg[1:]

        # calculate the real time stamp for each shots
        t_spokes = acq_t0 + np.arange(n_spokes) * TR
        idx_spokes = np.arange(n_spokes)

        target_RR = np.mean(np.diff(ecg))

        # spokes aligned in one RR interval
        idx_aligned_spokes = []
        t_aligned_spokes = []
        # loop for each RR interval to find the shots in that heart beat period
        for idx_ecg in range(min(ecg.size - 1, num_cycles)):
            tperiod_start = ecg[idx_ecg]
            tperiod_end = ecg[idx_ecg + 1]
            current_RR = tperiod_end - tperiod_start

            current_RR_spokes_mask = np.logical_and(t_spokes >= tperiod_start, t_spokes < tperiod_end)
            # current_RR_spokes_mask = np.logical_and(current_RR_spokes_mask, t_spokes >= Ndiscard)
            
            idx_current_RR_spokes = idx_spokes[current_RR_spokes_mask]
            t_current_RR_spokes = t_spokes[current_RR_spokes_mask]
            t_current_RR_spokes_relative = (t_current_RR_spokes - tperiod_start) / current_RR       # [0,1]
            idx_aligned_spokes.extend(idx_current_RR_spokes)
            # t_aligned_spokes are relative time stamp normalized to the [0,1]
            t_aligned_spokes.extend(t_current_RR_spokes_relative)
        
        # sort the time stamps and indexs of all alighened spokes in each RR interval
        (t_aligned_spokes, idx_aligned_spokes) = list(zip(*sorted(zip(t_aligned_spokes, idx_aligned_spokes))))
        
        aligned_data = {
            'target_RR': target_RR,
            'num_cycles': num_cycles,
            't_spokes': t_aligned_spokes,   # nSpokes
            'idx_spokes': idx_aligned_spokes
        }
        return aligned_data
    
    def get_subject_data(self, file_folder, num_cycles):
        case_name = file_folder.split('/')[-1]
        h5_data = h5py.File(f'{file_folder}/CINE_testrun.h5', 'r', libver='latest', swmr=True)

        # load coil sensitivity maps (`csm`)
        csm = h5_data['csm_real'][self.config['coil_select']] + 1j*h5_data['csm_imag'][self.config['coil_select']]

        # `full_kdata` is the continously acquired k-space data
        full_kdata = h5_data['fullkdata_real'][self.config['coil_select']] + \
            1j * h5_data['fullkdata_imag'][self.config['coil_select']]
        # `full_kpos` contains the position of the spokes in the k-space, i.e., trajectory
        full_kpos = h5_data['fullkpos'][()]

        # frequency padding for better boundary condition
        # if padding:
        #     nFE = full_kdata.shape[-1]
        #     n_fpad = int(0.15 * nFE)
        #     full_kdata = np.pad(full_kdata, ((0, 0), (0, 0), (n_fpad, n_fpad)), 'constant')
        #     # full_kdata = np.pad(full_kdata, ((0, 0), (0, 0), (n_fpad, n_fpad)), 'reflect')
        #     full_kpos_pad = np.pad(full_kpos, ((0, 0), (0, 0), (n_fpad, n_fpad)), 'constant')
        #     full_kpos_pad[:,:,0:n_fpad] = full_kpos[:,:,0:1] + full_kpos[:,:,nFE//2-n_fpad:nFE//2]
        #     full_kpos_pad[:,:,-n_fpad:] = full_kpos[:,:,nFE//2:nFE//2+n_fpad] + full_kpos[:,:,-1:]
        #     full_kpos = full_kpos_pad

        # # crop csm to img_dim
        # csm = medutils.visualization.center_crop(csm, (self.img_dim, self.img_dim))
        csm_rss = medutils.mri.rss(csm, coil_axis=0)
        csm = np.nan_to_num(csm/csm_rss)
        self.csm = csm

        # k-space dimensions
        nc, n_total_spokes, nFE = full_kdata.shape

        # patient-related binning info
        ecg = h5_data['ECG'][0]
        acq_t0 = h5_data['read_marker'][0][0] # original acq_t0

        # TR based on read_marker
        TR = (h5_data['read_marker'][0][-1] - h5_data['read_marker'][0][0] 
                + h5_data['read_marker'][0][1] - h5_data['read_marker'][0][0]) / n_total_spokes
        h5_data.close()
        
        # align spokes to one RR interval (heart beat)
        aligned_data = self.align_to_one_cycle(ecg, acq_t0, TR, n_total_spokes, num_cycles)
        

        # undersampled data
        t_spokes = aligned_data['t_spokes']
        idx_spokes = aligned_data['idx_spokes']
        target_RR = aligned_data['target_RR']

        # # padding for temporal dimension
        # if padding:
        #     n_tpad = int(0.15 * len(tshots))
        #     idxshots = list(reversed(idxshots[:n_tpad])) + idxshots + list(reversed(idxshots[-n_tpad:]))
        #     tshots = list(list(reversed(tshots[0] - tshots[:n_tpad])) + tshots[0]) + tshots + list(list(reversed(tshots[-1] - tshots[-n_tpad:])) + tshots[-1])


        kdata = full_kdata[:, idx_spokes, :] # nc, nSpokes, nFE

        ktraj = full_kpos[:, idx_spokes, :] * 2 # normalized to [-1, 1]
        ktraj = np.roll(ktraj, shift=1, axis=0) # 2, nSpokes, nFE

        # normalize k-space data、
        # TODO: normalize k-space data in a more reasonable way
        kdata = kdata / np.max(np.abs(kdata)) #* 255

        # self.n_kpoints = kdata.shape[-1] * kdata.shape[-2] * kdata.shape[-3] #nFE*nSpokes*nc
        
        data = {
            'caseid': case_name,
            'kdata': kdata, # nc, nSpokes, nFE
            'ktraj': ktraj, # 2, nSpokes, nFE
            'tspokes': np.array(t_spokes), # nSpokes        [0,1]
            'RR': target_RR,  # 1
            'csm': csm, 
        }
        
        return data
    
    # TODO: add method for traditional binning 
    # def get_binned_data(self, file_folder, num_cycles):
    #    pass

    def __len__(self):
        return self.n_kpoints

    def __getitem__(self, index):
        # point wise sampling
        sample = {
            'coords': self.kcoords_flat[index],
            'targets': self.kdata_flat[index]
        }
        return sample
