from random import sample
import numpy as np

def convert_to_tensor(batch):
    formated_batched = [np.stack(batch[i][0]) for i in range(len(batch))]