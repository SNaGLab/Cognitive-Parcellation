import numpy as np
import h5py


path = '../../data/weights/alexnet_weights.h5'

f = h5py.File(path, 'r')

# List all groups
#print("Keys: %s" % f.keys())

for key in f.keys():
	# Get the data
	print key
	for key2 in f[key].keys():
		print f[key][key2]


f.close()
