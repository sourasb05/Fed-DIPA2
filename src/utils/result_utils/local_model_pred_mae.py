import glob
import torch
import sys

if __name__ == "__main__":
    MODELS_DIR = "./models/FedDcprivacy/local_model/"
    print(MODELS_DIR)
    for model_path in glob.glob(MODELS_DIR + "*"):
        model = torch.load(model_path + "/best_local_checkpoint.pt")
        print(model)
        sys.exit()