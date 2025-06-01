from torch.utils.data import Dataset
import h5py
import torch

class SNPsDataset_HDF5(Dataset):
    def __init__(self, hdf5_filename: str, preload=True):
        self.hdf5_filename = hdf5_filename
        self.preload = preload
        self.data_file = None  # For lazy loading

        if self.preload:
            with h5py.File(self.hdf5_filename, 'r') as f:
                self.snps = torch.from_numpy(f['snps'][:]).long()
                self.snpsIndex = torch.from_numpy(f['snpsIndex'][:]).long()
            self.length = len(self.snps)
        else:
            with h5py.File(self.hdf5_filename, 'r') as f:
                self.length = len(f['snps'])

    def __len__(self):
        return self.length

    def __getitem__(self, idx):
        if self.preload:
            return self.snps[idx], self.snpsIndex[idx]

        if self.data_file is None:
            self.data_file = h5py.File(self.hdf5_filename, 'r')

        snps = torch.from_numpy(self.data_file['snps'][idx]).long()
        snpsIndex = torch.from_numpy(self.data_file['snpsIndex'][idx]).long()
        return snps, snpsIndex

    def __del__(self):
        if hasattr(self, 'data_file') and self.data_file is not None:
            self.data_file.close()

    def close(self):
        if hasattr(self, 'data_file') and self.data_file is not None:
            self.data_file.close()
            self.data_file = None
        if hasattr(self, 'snps'):
            del self.snps
        if hasattr(self, 'snpsIndex'):
            del self.snpsIndex
