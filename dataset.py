import numpy as np
import torch
from torch.utils.data import Dataset

class SyntheticSpikeTrains(Dataset):
    def __init__(self, T, N, class1='normal',class2='normal'):
        self.N = N
        self.T = T
        self.class1 = class1
        self.class2 = class2
        self.N_FEATURES = 1
        self.N_CLASSES = 2
        self.data, self.labels = self.generate()
        self.train_ix, self.test_ix = self.getSplitIndices()
       
        
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, ix):
        return self.data[ix], self.labels[ix]
    
    def getSplitIndices(self):
        split_props = [0.8, 0.2]
        indices = np.arange(self.N)
        split_points = [int(self.N*i) for i in split_props]
        train_ix = np.random.choice(indices,
                                    split_points[0],
                                    replace=False)
        test_ix = np.random.choice((list(set(indices)-set(train_ix))),
                                   split_points[1],
                                   replace=False)
        return train_ix, test_ix
    
    def generate(self):
        """
        signals generate depending on diff distributions
        1. normal : p=[.001,.001,.024,.154,.3, .35, .154, .014,.001,.001]
        2. uniform : p = [.1, .1, .1, .1, .1, .1, .1, .1, .1, .1]
        3. left-skewed : p = [.6,.35,.02,.02,.004, .002, .001, .001,.001,.001]
        4. right-skewed : p = [.001, .001, .001, .001, .002, .004, .02, .02, .35, .6]
        """
        normal = [.001,.001,.024,.154,.3, .35, .154, .014,.001,.001]
        uni = [.1, .1, .1, .1, .1, .1, .1, .1, .1, .1]
        left = [.6,.35,.02,.02,.004, .002, .001, .001,.001,.001]
        right = [.001, .001, .001, .001, .002, .004, .02, .02, .35, .6]
        x = np.zeros((self.N, self.T, 1))
        y = np.random.randint(self.N_CLASSES, size=int(self.N))
        
        for i in range(int(self.N)):
            if y[i] == 0: # rate : 0.6
                if self.class1 == 'normal':
                    signal_locs = np.random.choice(np.arange(0, 10), size = 6, replace=False,p=normal)
                elif self.class1 == 'uniform':
                    signal_locs = np.random.choice(np.arange(0, 10), size = 6, replace=False,p=uni)
                elif self.class1 == 'left':
                    signal_locs = np.random.choice(np.arange(0, 10), size = 6, replace=False,p=left)
                elif self.class1 == 'right':
                    signal_locs = np.random.choice(np.arange(0, 10), size = 6, replace=False,p=right)
                x[i, signal_locs,0] = 1
            else: # rate : 0.2
                if self.class2 == 'normal':
                    signal_locs = np.random.choice(np.arange(0, 10), size = 2, replace=False,p=normal)
                elif self.class2 == 'uniform':
                    signal_locs = np.random.choice(np.arange(0, 10), size = 2, replace=False,p=uni)
                elif self.class2 == 'left':
                    signal_locs = np.random.choice(np.arange(0, 10), size = 2, replace=False,p=left)
                elif self.class2 == 'right':
                    signal_locs = np.random.choice(np.arange(0, 10), size = 2, replace=False,p=right)
                x[i, signal_locs, 0] = 1
        data = torch.tensor(np.asarray(x).astype(np.float32),
                            dtype=torch.float)
        labels = torch.tensor(np.array(y).astype(np.int32), dtype=torch.long)
        
        return data, labels
        