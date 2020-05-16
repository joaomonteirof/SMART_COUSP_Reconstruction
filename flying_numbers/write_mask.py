import numpy as np

mask = np.random.rand(64,64)#.round()

np.save('./mask.npy', mask)