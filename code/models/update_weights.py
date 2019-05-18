import numpy as np
import h5py


path = '../../data/weights/alexnet_weights.h5'

f = h5py.File(path, 'r')

# List all groups
print("Keys: %s" % f.keys())
a_group_key = list(f.keys())[0]

# Get the data
data = f[a_group_key].values()
print data[0], data[1]
f.close()
