import numpy as np

mask = np.random.rand(256,256).round()

print(mask)

np.save('./mask.npy', mask)