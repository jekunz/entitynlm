import torch
import os
from torch.utils import data

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

class CorpusLoader(data.Dataset):
    'Characterizes a dataset for PyTorch'
    def __init__(self, data_dir='data', partition='train', ID_store={}, vocab={}, gvocab={}, auto_load = True, verbose = True):
        #super().__init__()
        'Initialization'
        self.ID_store = ID_store
        self.data_dir = data_dir
        self.partition = partition
        self._vocab = vocab
        self._gvocab = gvocab
        self.verbose = verbose
        if auto_load:
            self.load_file_IDs()
    
    def load_file_IDs(self, id_file='file_ids.dict'):
        with open(f'{self.data_dir}/{id_file}') as f:
            self.ID_store = eval(f.read())
        if self.verbose:
            print(f"{len(self)} file IDs (documents) for partition '{self.partition}' succesfully loaded.")

        with open(f'{self.data_dir}/vocab.dict') as f:
            self._vocab = eval(f.read())
        if self.verbose:
            print(f"Vocab dict with vocabulary of {len(self._vocab)} tokens loaded.")
        if os.path.isfile(f'{self.data_dir}/gvocab.dict'):
            with open(f'{self.data_dir}/gvocab.dict') as f:
                self._gvocab = eval(f.read())
            if self.verbose:
                print(f"Extended gvocab dict with vocabulary of {len(self._gvocab)} tokens loaded.")

    @property
    def vocab(self):
    	return max((self._vocab, self._gvocab), key=len)

    def __len__(self):
        'Denotes the total number of samples'
        return len(self.ID_store[self.partition])

    def __getitem__(self, index):
        'Generates one sample of data'
        # Select sample
        ID = self.ID_store[self.partition][index]

        # Load data and get label
        X = torch.load(f'{self.data_dir}/{self.partition}/doc_{ID:03}_tokens.pt') #map_location=lambda storage, loc: storage.cuda(0))
        y = torch.load(f'{self.data_dir}/{self.partition}/doc_{ID:03}_entities.pt')
        #map_location=lambda storage, loc: storage.cuda(0))
        z = torch.zeros(y.size(0), dtype=torch.float)
        z[y > 0] = 1
        z = z.contiguous()
        return X, y, z




