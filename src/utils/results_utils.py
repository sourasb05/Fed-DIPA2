import h5py
import numpy as np
import pandas as pd 
import os
import numpy as np
import re
import csv
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1.inset_locator import inset_axes


def convert_csv_to_txt(input_file,output_file):
   
    with open(input_file, 'r') as csv_file, open(output_file, 'w') as space_delimited_file:
        csv_reader = csv.reader(csv_file)
        for row in csv_reader:
            space_delimited_file.write(' '.join(row) + '\n')

    print(f'CSV file "{input_file}" converted to space-delimited file "{output_file}"')



def read_file(file):
    hf = h5py.File(file, 'r')
    attributes = []
    for key in hf.keys():
        attributes.append(key)
    
    return attributes, hf


def get_data(hf,attributes):
    data = []
    pm = []
    acc_pm = []
    loss_pm = []
    loss_gm = []
    for i in range(len(attributes)):
        ai = hf.get(attributes[i])
        ai = np.array(ai)
        data.append(ai)
    
    return data

def convergence_analysis(path, acc_file, loss_file):
    dir_list = os.listdir(path)
    
    Fedavg_test_loss = []
    Fedavg_test_accuracy = []
    Fedavg_train_loss = []
    Fedavg_train_accuracy = []

    
    Fedprox_test_loss = []
    Fedprox_test_accuracy = []
    Fedprox_train_loss = []
    Fedprox_train_accuracy = []

    Fedmem_test_loss = []
    Fedmem_test_accuracy = []
    Fedmem_train_loss = []
    Fedmem_train_accuracy = []

    Fedse_test_loss = []
    Fedse_test_accuracy = []
    Fedse_train_loss = []
    Fedse_train_accuracy = []

    Demlearn_test_loss = []
    Demlearn_test_accuracy = []
    Demlearn_train_loss = []
    Demlearn_train_accuracy = []


    for file_name in dir_list:
        if file_name in ['fedavg.h5','fedprox.h5', 'fedmem.h5', 'fedse.h5', 'demlearn.h5', 'hsgd.h5']:
            print(file_name)
            attributes, hf = read_file(path+file_name)

            data = get_data(hf,attributes)
            #id=0
            for key in hf.keys():
                attributes.append(key)
                # print("id [",id,"] :", key)
                #id+=1
            if file_name in ['fedavg.h5', 'fedprox.h5','hsgd.h5']:    
                train_loss = hf.get('global_train_loss')
                train_acc = hf.get('global_train_accuracy')   
                val_loss = hf.get('global_test_loss')
                val_acc = hf.get('global_test_accuracy')

                if file_name == "fedavg.h5":
                    Fedavg_train_loss.append(np.array(train_loss).tolist())
                    Fedavg_train_accuracy.append(np.array(train_acc).tolist())
                    Fedavg_test_loss.append(np.array(val_loss).tolist())
                    Fedavg_test_accuracy.append(np.array(val_acc).tolist())

                    print(Fedavg_train_loss)
                elif file_name == "fedprox.h5":

                    Fedprox_train_loss.append(np.array(train_loss).tolist())
                    Fedprox_train_accuracy.append(np.array(train_acc).tolist())
                    Fedprox_test_loss.append(np.array(val_loss).tolist())
                    Fedprox_test_accuracy.append(np.array(val_acc).tolist())
            
            
            elif file_name in ['fedmem.h5', 'fesem.h5','demlearn.h5']:        
                train_loss = hf.get('per_train_loss')
                train_acc = hf.get('per_train_accuracy')   
                val_loss = hf.get('per_test_loss')
                val_acc = hf.get('per_test_accuracy')
                
                if file_name == "fedmem.h5":

                    Fedmem_train_loss.append(np.array(train_loss).tolist())
                    Fedmem_train_accuracy.append(np.array(train_acc).tolist())
                    Fedmem_test_loss.append(np.array(val_loss).tolist())
                    Fedmem_test_accuracy.append(np.array(val_acc).tolist())

            
    train_loss = {
        'GR' : np.arange(50),
        'FedAvg' :   Fedavg_train_loss[0][:],
        'Fedprox' : Fedprox_train_loss[0][:],
        'Fedmem' : Fedmem_train_loss[0][:],
    }


    train_acc = {
        'GR' : np.arange(50),
        'FedAvg' :   Fedavg_train_accuracy[0][:],
        'Fedprox' : Fedprox_train_accuracy[0][:],
        'Fedmem' : Fedmem_train_accuracy[0][:],
    }


    val_loss = {
        'GR' : np.arange(50),
        'FedAvg' :   Fedavg_test_loss[0][:],
        'Fedprox' : Fedprox_test_loss[0][:],
        'Fedmem' : Fedmem_test_loss[0][:],
    }


    val_acc = {
        'GR' : np.arange(50),
        'FedAvg' :   Fedavg_test_accuracy[0][:],
        'Fedprox' : Fedprox_test_accuracy[0][:],
        'Fedmem' : Fedmem_test_accuracy[0][:],
    }

    
    df_train_loss = pd.DataFrame(train_loss)
    df_train_acc = pd.DataFrame(train_acc)
    df_val_loss = pd.DataFrame(val_loss)
    df_val_acc = pd.DataFrame(val_acc)

    csv_train_acc_path = path + "train_" + acc_file +".csv"
    csv_train_loss_path = path + "train_" + loss_file +".csv"
    csv_val_acc_path = path + "test_" + acc_file +".csv"
    csv_val_loss_path = path + "test_" +loss_file +".csv"
    
    
    txt_train_acc_path = path + "train_" + acc_file +".txt"
    txt_train_loss_path = path +  "train_" + loss_file +".txt"
    txt_val_acc_path = path + "test_" + acc_file +".txt"
    txt_val_loss_path = path + "test_" + loss_file +".txt"

    df_train_acc.to_csv(csv_train_acc_path, index=False)
    df_train_loss.to_csv(csv_train_loss_path, index=False)
    df_val_acc.to_csv(csv_val_acc_path, index=False)
    df_val_loss.to_csv(csv_val_loss_path, index=False)
    
    convert_csv_to_txt(csv_train_acc_path,txt_train_acc_path)
    convert_csv_to_txt(csv_train_loss_path,txt_train_loss_path)
    convert_csv_to_txt(csv_val_acc_path,txt_val_acc_path)
    convert_csv_to_txt(csv_val_loss_path,txt_val_loss_path)

    plot_convergence(Fedavg_test_loss[0],
                    Fedavg_test_accuracy[0],
                    Fedavg_train_loss[0],
                    Fedavg_train_accuracy[0],
                    Fedprox_test_loss[0],
                    Fedprox_test_accuracy[0],
                    Fedprox_train_loss[0],
                    Fedprox_train_accuracy[0],
                    Fedmem_test_loss[0],
                    Fedmem_test_accuracy[0],
                    Fedmem_train_loss[0],
                    Fedmem_train_accuracy[0],
                    path)

def plot_convergence( Fedavg_test_loss,
                    Fedavg_test_accuracy,
                    Fedavg_train_loss,
                    Fedavg_train_accuracy,
                    Fedprox_test_loss,
                    Fedprox_test_accuracy,
                    Fedprox_train_loss,
                    Fedprox_train_accuracy,
                    Fedmem_test_loss,
                    Fedmem_test_accuracy,
                    Fedmem_train_loss,
                    Fedmem_train_accuracy,
                    path):
        
        
        fig, ax = plt.subplots(1,4, figsize=(20,4))

        ax[0].plot(Fedavg_test_loss, label= "FedAvg")
        ax[0].plot(Fedprox_test_loss, label= "FedProx")
        ax[0].plot(Fedmem_test_loss, label= "FedMEM(PM)")
        
        ax[0].set_xlabel("Global Iteration")
        #ax[0].set_xscale('log')
        ax[0].set_ylabel("Validation Loss")
        #ax[0].set_yscale('log')
        ax[0].set_xticks(range(0, 50, int(50/5)))
        #ax[0].legend(prop={"size":12})
        #ax[0].legend()
        #x1, x2, y1, y2 = 600, 800, 0.15, 2.0  # Adjust these values as needed
        #axins = inset_axes(ax[0], width="50%", height="50%", loc=7)
        #axins.plot(Fedavg_test_loss)
        #axins.plot(Fedprox_test_loss) 
        #axins.plot(Fedmem_test_loss) 
        #axins.set_xlim(x1, x2)
        #axins.set_ylim(y1, y2)
        #axins.indicate_inset_zoom(axins, edgecolor="black")

       # ax[1].plot(Fedavg_gd_test_accuracy, label= "FedAvg+GD")
        #ax[1].plot(Fedavg_sgd_test_accuracy, label= "FedAvg+SGD")
        ax[1].plot(Fedavg_test_accuracy, label= "FedAvg")
        ax[1].plot(Fedprox_test_accuracy, label= "FedProx")
        ax[1].plot(Fedmem_test_accuracy, label= "FedMEM(PM)")
        ax[1].set_xlabel("Global Iteration")
        ax[1].set_xticks(range(0, 50, int(50/5)))
        ax[1].set_ylabel("Validation Accuracy")
        #ax[1].legend(prop={"size":12})
        #ax[1].legend()
        """x1, x2, y1, y2 = 600, 800, 0.8, 0.92  # Adjust these values as needed
        axins = inset_axes(ax[1], width="50%", height="50%", loc=7)
        axins.plot(Fedavg_test_accuracy)
        axins.plot(Fedprox_test_accuracy) 
        axins.plot(Fedmem_test_accuracy) 
        axins.set_xlim(x1, x2)
        axins.set_ylim(y1, y2)
        axins.indicate_inset_zoom(axins, edgecolor="black")
        """
        ax[2].plot(Fedavg_train_loss , label= "FedAvg")
        ax[2].plot(Fedprox_train_loss, label= "FedProx")
        ax[2].plot(Fedmem_train_loss, label= "FedMEM(PM)")
        ax[2].set_xlabel("Global Iteration")
        #ax[2].set_xscale('log')
        ax[2].set_ylabel("Training Loss")
        #ax[2].set_yscale('log')
        ax[2].set_xticks(range(0, 50, int(50/5)))
        #ax[2].legend(prop={"size":12})
        #ax[2].legend()
        """x1, x2, y1, y2 = 600, 800, 0.1, 0.55  # Adjust these values as needed
        axins = inset_axes(ax[2], width="50%", height="50%", loc=7)
        axins.plot(Fedavg_train_loss, label="FedAvg")
        axins.plot(Fedprox_train_loss, label= "FedProx")
        axins.plot(Fedmem_train_loss, label= "FedMEM(PM)")
        axins.set_xlim(x1, x2)
        axins.set_ylim(y1, y2)
        axins.indicate_inset_zoom(axins, edgecolor="black")
        """
        # ax[3].plot(Fedavg_gd_train_accuracy, label= "FedAvg+GD")
        #ax[3].plot(Fedavg_sgd_train_accuracy, label= "FedAvg+SGD")
        ax[3].plot(Fedavg_train_accuracy, label= "FedAvg")
        ax[3].plot(Fedprox_train_accuracy, label= "FedProx")
        ax[3].plot(Fedmem_train_accuracy, label= "FedMEM(PM)")
        ax[3].set_xlabel("Global Iteration")
        ax[3].set_ylabel("Training Accuracy")
        ax[3].set_xticks(range(0, 50, int(50/5)))
        #ax[3].legend(prop={"size":12})
        #ax[3].legend()

        """
        x1, x2, y1, y2 = 600, 800, 0.8, 0.98  # Adjust these values as needed
        axins = inset_axes(ax[3], width="50%", height="50%", loc=7)
        axins.plot(Fedavg_train_accuracy)
        axins.plot(Fedprox_train_accuracy) 
        axins.plot(Fedmem_train_accuracy) 
        axins.set_xlim(x1, x2)
        axins.set_ylim(y1, y2)
        axins.indicate_inset_zoom(axins, edgecolor="black")
        """
        handles, labels = ax[3].get_legend_handles_labels()
        fig.legend(handles, labels, loc='upper center', ncol=6, fontsize=12)


        plt.draw()
       
        plt.savefig(path +'convergence.png')

        # Show the graph
        plt.show()



def average_result(path,directory_name, algorithm, avg_file):
    
    dir_list = os.listdir(path)

    i=0
    train_loss, train_accuracy, test_loss, test_accuracy = [], [], [], []

    if algorithm in [ "Fedmem", "", "FedavgPGD", "FedavgPSGD", "Fedprox", "Fedfw", "Feddr" ]:
        for file_name in dir_list:
            
            
            if file_name.endswith(".h5"):
                print(file_name)
                attributes, hf = read_file(path+file_name)

                data = get_data(hf,attributes)
                id=0
                for key in hf.keys():
                    attributes.append(key)
                    # print("id [",id,"] :", key)
                    id+=1

                gtsl = hf.get('global_test_loss')
                gtrl = hf.get('global_train_loss')
                gtsa = hf.get('global_test_accuracy')
                gtra = hf.get('global_train_accuracy')

                test_loss.append(np.array(gtsl).tolist())
                train_loss.append(np.array(gtrl).tolist())
                test_accuracy.append(np.array(gtsa).tolist())
                train_accuracy.append(np.array(gtra).tolist())
   
                
            
        avg_train_loss = np.array(train_loss)
        avg_test_loss = np.array(test_loss)
        avg_train_accuracy = np.array(train_accuracy)
        avg_test_accuracy = np.array(test_accuracy)

        # print(avg_test_accuracy)
        
        gtrl_mean = np.mean(avg_train_loss, axis=0)
        
        gtra_mean = np.mean(avg_train_accuracy, axis=0)
        # print(gtra_mean)
        gtsl_mean = np.mean(avg_test_loss, axis=0)
        gtsa_mean = np.mean(avg_test_accuracy, axis=0)

        gtrl_std = np.std(avg_train_loss, axis=0)
        gtra_std = np.std(avg_train_accuracy, axis=0)
        gtsl_std = np.std(avg_test_loss, axis=0)
        gtsa_std = np.std(avg_test_accuracy, axis=0)

        gtrl_mean_std = np.column_stack((gtrl_mean, gtrl_std))
        gtra_mean_std = np.column_stack((gtra_mean, gtra_std))
        gtsl_mean_std = np.column_stack((gtsl_mean, gtsl_std))
        gtsa_mean_std = np.column_stack((gtsa_mean, gtsa_std))

        training_loss_mean_std = gtrl_mean_std[gtrl_mean_std[:,0].argmin()]
        training_acc_mean_std = gtra_mean_std[gtra_mean_std[:,0].argmax()]
        val_loss_mean_std = gtsl_mean_std[gtsl_mean_std[:,0].argmin()]
        val_acc_mean_std = gtsa_mean_std[gtsa_mean_std[:,0].argmax()]

        
        with h5py.File(directory_name  + '{}.h5'.format(avg_file), 'w') as hf:
            hf.create_dataset('avg_training_loss', data=gtrl_mean)
            hf.create_dataset('avg_training_accuracy', data=gtra_mean)
            hf.create_dataset('avg_test_loss', data=gtsl_mean)
            hf.create_dataset('avg_test_accuracy', data=gtsa_mean)
            hf.close



        print("+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-")
        print("Algorithm :",algorithm)
        print("Global training loss (mean/std) : (",training_loss_mean_std[0],"/",training_loss_mean_std[1],")")
        print("Global training accuracy (mean/std) : (",training_acc_mean_std[0],"/",training_acc_mean_std[1],")")
        print("Global test loss (mean/std) : (", val_loss_mean_std[0],"/", val_loss_mean_std[1],")")
        print("Global test accuracy (mean/std) : (",val_acc_mean_std[0],"/",val_acc_mean_std[1],")")
        print("+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-\n")
        

"""
Get the average of accuracy and loss
"""

"""path = "/proj/sourasb-220503/codebase/FedFWplus/results/FedFW/SYNTHETIC/MCLR/time_varing_eta/time_varing_lambda/perf/10/"
directory_name = "/proj/sourasb-220503/codebase/FedFWplus/results/convergence/SYNTHETIC/10/mclr/"
average_result(path, directory_name, 'Fedfw', 'fedfw')
"""

path = "/proj/sourasb-220503/FedMEM/results/convergence/"
acc_file = "accuracy_sota"
loss_file = "loss_sota"

convergence_analysis(path, acc_file, loss_file)
