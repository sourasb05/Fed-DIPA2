import zipfile

# Specify the path to your zip file
zip_file_path = '/proj/sourasb-220503/FedMEM/dataset/r3_mem_ResNet50_features.zip'

# Specify the directory where you want to extract the zip file
extract_to_directory = '/proj/sourasb-220503/FedMEM/dataset/'

# Open the zip file in read mode
with zipfile.ZipFile(zip_file_path, 'r') as zip_ref:
    # Extract all the contents into the directory specified
    zip_ref.extractall(extract_to_directory)

print(f'Extracted all contents of {zip_file_path} to {extract_to_directory}')