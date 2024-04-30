import h5py
import json
#  Read JSON data from an HDF5 file
with h5py.File('./results/FedAvg/h5/global_model/exp_no_0_GR_5_BS_64.h5', 'r') as hdf:
    read_data = json.loads(hdf['global_test_metric'][0].decode('utf-8'))
    
# Verify the data
for i in range(5):
    print(read_data[i]['Accuracy'])