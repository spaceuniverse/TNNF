__author__ = 'rhrub'


import csv
import numpy as np
import h5py

srcFolder = './src/'
csv_type = '.csv'
hdf_type = '.hdf5'
target_csv = 'mnist_test'
target_hdf = 'mnist_test'

#Get DATA from CSV
count = 0
array = []
with open(srcFolder + target_csv + csv_type, 'rb') as f:
    reader = csv.reader(f)
    for row in reader:
        array.append(map(int, row))
        count += 1
        print count

#Convert to numpy array
array = np.array(array)

#Print out array's shape and a few values for quick verification
print array.shape, '\n', array[:3, :]

#Create HDF5 file
f = h5py.File(srcFolder + target_hdf + hdf_type, 'w')
hdfData = f.create_dataset('hdfDataSet', shape=array.shape)

#Fill in HDF with data
hdfData[:, :] = array[:, :]

#Print out HDF's shape and a few values for quick verification with array above. Should be the same.
print hdfData.shape, '\n', hdfData[:3, :]
f.close()
