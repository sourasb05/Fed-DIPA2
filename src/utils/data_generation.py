import os
from PIL import Image
import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
import csv
import numpy as np
import random
from tqdm import trange
import json
from sklearn.model_selection import train_test_split
import random
import time

class MyDataset(Dataset):
    def __init__(self, data_dir, csv_file, transform):
        self.data_dir = data_dir
        self.csv_file = csv_file
        self.transform = transform
        
        self.data = []
        self.target = []

        # Load the labels from the CSV file
        with open(csv_file, "r") as f:
            next(f)
            reader = csv.reader(f)
            for row in reader:
                self.data.append(os.path.join(data_dir, row[0]))
                self.target.append(int(row[1]))
           
           
           # self.labels = [line.strip().split(",")[1] for line in f]
            # print(self.labels)
    

    def __len__(self):
        print("number of labels:",len(self.labels))

        return len(self.data)

    def __getitem__(self, idx):
        image = Image.open(self.data[idx])
        if self.transform is not None:
            image = self.transform(image)
        
        # Convert the image to a PyTorch tensor
        # image = torch.from_numpy(np.array(image))

        return image, self.target[idx]


"""class MyDataset(Dataset):
    def __init__(self, data_dir, csv_file): # .transform
        self.data_dir = data_dir
        self.csv_file = csv_file
        # self.transform = transform
        
        self.data = []
        self.target = []

        # Load the labels from the CSV file
        with open(csv_file, "r") as f:
            next(f)
            reader = csv.reader(f)
            for row in reader:
                self.data.append(os.path.join(data_dir, row[0]))
                self.target.append(int(row[1]))
            # print(self.data)
            # print(self.target)
           
           
           # self.labels = [line.strip().split(",")[1] for line in f]
            # print(self.labels)
    

    def __len__(self):
        print("number of labels:",len(self.labels))

        return len(self.data)

    def __getitem__(self, idx):
        image = torch.load(self.data[idx])
        #if self.transform is not None:
        #    image = self.transform(image)
        
        # Convert the image to a PyTorch tensor
        # image = torch.from_numpy(np.array(image))

        return image, self.target[idx]
"""
def generate_list_with_sum(n, total_sum, min_value, max_value):
    """Generate a list of 'n' integers with a specific total sum and within a given range."""
    # Check if it's possible to create such a list
    if n * min_value > total_sum or n * max_value < total_sum:
        return None
    
    elements = []
    remaining_sum = total_sum
    for i in range(n - 1):
        max_possible_value = min(max_value, remaining_sum - (n - i - 1) * min_value)
        min_possible_value = max(min_value, remaining_sum - (n - i - 1) * max_value)
        value = random.randint(min_possible_value, max_possible_value)
        elements.append(value)
        remaining_sum -= value

    # Add the last element
    elements.append(remaining_sum)

    return elements

def load_data(current_directory, data_div):
    print("at load data")
    # data_dir = current_directory + "/dataset/r3_mem_ResNet50FC_features"
    # csv_file = current_directory + "/dataset/tensor_training_file.csv"
    
    data_dir = current_directory + "/dataset/r3_refined"
    csv_file = current_directory + "/dataset/refined_training_file.csv"
    
    # Create a transform to resize and normalize the images
    transform = transforms.Compose([
        transforms.Resize(32),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ])
    # Create a Dataset object
    dataset = MyDataset(data_dir, csv_file, transform)

    images = []
    targets = []
    for image, target in dataset:
        images.append(image.cpu().detach().numpy())
        targets.append(target)

    #print(images)
    #input("press")
    #print(targets)
    
    """for i, image in enumerate(images):
        if isinstance(image, torch.Tensor):  # Check if it's a PyTorch tensor
            print(f"images[{i}] is a tensor.")
        else:
            print(f"images[{i}] is not a tensor.")
    """
    # input("press")
    n = len(images)
    print("number of images:", n)

    if len(images) == len(targets):
        print(" images have equivalent tagets")

    random.seed(1)
    np.random.seed(1)
    NUM_USERS = 20

    # Setup directory for train/test data
    train_path = current_directory +'/FedMEM_dataset/train/equal_from_img_fedr3_train.json'
    test_path = current_directory +'/FedMEM_dataset/test/equal_from_img_fedr3_test.json'
    dir_path = os.path.dirname(train_path)
    if not os.path.exists(dir_path):
       os.makedirs(dir_path)
    dir_path = os.path.dirname(test_path)
    if not os.path.exists(dir_path):
        os.makedirs(dir_path)
    
    if data_div == "equal":
        
        m = np.ones(NUM_USERS, dtype=int) * int(n/NUM_USERS)
        print(m)

        X = [[] for _ in range(NUM_USERS)]
        y = [[] for _ in range(NUM_USERS)]
        lb=0
        i = 0
        for user in trange(NUM_USERS):
            print("lb :",lb)
            print("lb+m[user]: ",)
            # X[user] += images[lb:lb+520].tolist()
            X[user] += [image.tolist() for image in images[lb:lb+m[user]]]
            y[user] += targets[lb:lb+m[user]]
            lb = lb+m[user]

    elif data_div == "unequal": 
        if NUM_USERS < 2:
            raise ValueError("Size must be greater than 1.")

        unequal_parts = generate_list_with_sum(NUM_USERS,n, min_value=500, max_value=4000)

        
        print(unequal_parts)
        input("press")
        X = [[] for _ in range(NUM_USERS)]
        y = [[] for _ in range(NUM_USERS)]
        lb=0
        i = 0

        for user in trange(NUM_USERS):
            print("lb :",lb)
            print("lb+unequal_parts[user]: ",lb+unequal_parts[user])
            # X[user] += images[lb:lb+520].tolist()
            X[user] += [image.tolist() for image in images[lb:lb+unequal_parts[user]]]
            y[user] += targets[lb:lb+unequal_parts[user]]
            lb = lb+unequal_parts[user]
        
    elif data_div == "n_iid": 
        print("at n_iid div")
        NUM_LABELS = 2
        if NUM_USERS < 2:
            raise ValueError("Size must be greater than 2.")
        data = []
        print(len(targets))
        #input("press")
        for i in trange(10):
            idx = targets == i
            print(len(images[idx]))
            input("press")
            data.append(images[idx])
            print(len(data[i]))
            #input("press")

        print("\nNumb samples of each label:\n", [len(v) for v in data])
        users_lables = []


        ###### CREATE USER DATA SPLIT #######
        # Assign 100 samples to each user
        X = [[] for _ in range(NUM_USERS)]
        y = [[] for _ in range(NUM_USERS)]
        idx = np.zeros(10, dtype=np.int64)
        for user in range(NUM_USERS):
            for j in range(NUM_LABELS):  # 2 labels for each users
                #l = (2*user+j)%10
                l = (user + j) % 10
                print("L:", l)
                X[user] += data[l][idx[l]:idx[l]+10].tolist()
                y[user] += (l*np.ones(10)).tolist()
                idx[l] += 10

        print("IDX1:", idx)  # counting samples for each labels

        # Assign remaining sample by power law
        user = 0
        props = np.random.lognormal(
            0, 2., (10, NUM_USERS, NUM_LABELS))  # last 5 is 5 labels
        props = np.array([[[len(v)-NUM_USERS]] for v in data]) * \
            props/np.sum(props, (1, 2), keepdims=True)
        # print("here:",props/np.sum(props,(1,2), keepdims=True))
        #props = np.array([[[len(v)-100]] for v in mnist_data]) * \
        #    props/np.sum(props, (1, 2), keepdims=True)
        #idx = 1000*np.ones(10, dtype=np.int64)
        # print("here2:",props)
        for user in trange(NUM_USERS):
            for j in range(NUM_LABELS):  # 4 labels for each users
                # l = (2*user+j)%10
                l = (user + j) % 10
                #num_samples = int(props[l, user//int(NUM_USERS/10), j])
                numran1 = random.randint(100, 250)
                num_samples = numran1  # + num_samples + numran1 #+ 200
                X[user] += data[l][idx[l]:idx[l]+num_samples].tolist()
                y[user] += (l*np.ones(num_samples)).tolist()
                idx[l] += num_samples
                print("check len of user:", user, "label:", j, "len data", len(X[user]), num_samples)

        print("IDX2:", idx) # counting samples for each labels




    # Create data structure
    train_data = {'users': [], 'user_data':{}, 'num_samples':[]}
    test_data = {'users': [], 'user_data':{}, 'num_samples':[]}
    all_samples=[]

    # Setup 5 users
    # for i in trange(5, ncols=120):
    # Setup 5 users
    # for i in trange(5, ncols=120):
    for i in range(NUM_USERS):
        uname = i
        X_train, X_test, y_train, y_test = train_test_split(X[i], y[i], train_size=0.7, stratify=y[i])
        # X_train, X_test, y_train, y_test = train_test_split(X[i], y[i], train_size=0.75, stratify=y[i])
        num_samples = len(X[i])
        #train_len = int(0.8*num_samples)  #Test 80%
        #test_len = num_samples - train_len


        train_data["user_data"][uname] = {'x': X_train, 'y': y_train}
        train_data['users'].append(uname)
        train_data['num_samples'].append(len(y_train))
    
        test_data['users'].append(uname)
        test_data["user_data"][uname] = {'x': X_test, 'y': y_test}
        test_data['num_samples'].append(len(y_test))
        # all_samples.append(train_len + test_len)

    print("Total_samples:",sum(train_data['num_samples'] + test_data['num_samples']))
    print("training_samples:", train_data['num_samples'])
    print("Total_testing_samples:",sum(train_data['num_samples']))
    print("testing_samples:", test_data['num_samples'])
    print("Total_testing_samples:",sum(test_data['num_samples']))
    # print("Median of data samples:", np.median(all_samples))




    # with open(train_path,'w') as outfile:
    #    json.dump(train_data, outfile)
    # with open(test_path, 'w') as outfile:
    #    json.dump(test_data, outfile)

    print("Finish Generating Samples")

    return train_data, test_data