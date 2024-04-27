import h5py
import json
#  Read JSON data from an HDF5 file
with h5py.File('/home/sourasb/PHD/Fed-DIPA2/results/FedAvg/h5/global_model/exp_no_0_GR_5_BS_64.h5', 'r') as hdf:
    read_data = json.loads(hdf['global_test_metric'][0].decode('utf-8'))

# Verify the data
print(read_data)