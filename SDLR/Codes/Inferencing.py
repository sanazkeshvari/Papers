import numpy as np
import torch
import os
import time

import sys
#sys.path.append(os.getcwd()[:os.getcwd().rfind("\\") + 1])
sys.path = [os.getcwd()[:os.getcwd().rfind("\\") + 1]] + sys.path

from allrank.data.dataset_loading import *
from allrank.training.train_utils import *

from functools import partial
import allrank.models.losses as losses
# from allrank.config import Config
from allrank.data.dataset_loading import load_libsvm_dataset, create_data_loaders
from torch.utils.data import DataLoader, Dataset
# from allrank.models.model import make_model
# from allrank.models.model_utils import get_torch_device, CustomDataParallel
# from allrank.training.train_utils import fit
# from allrank.utils.command_executor import execute_command
# from allrank.utils.experiments import dump_experiment_result, assert_expected_metrics
# from allrank.utils.file_utils import create_output_dirs, PathsContainer, copy_local_to_gs
# from allrank.utils.ltr_logging import init_logger
# from allrank.utils.python_utils import dummy_context_mgr



# import shutil
# import pandas as pd
#CUDA_LAUNCH_BLOCKING= 1
#os.environ["CUDA_LAUNCH_BLOCKING"] = 1
#os.system("set CUDA_LAUNCH_BLOCKING=1")

Model_Path = os.getcwd().replace("\\", "/") + "/../../Inferencing/Model_MQ2007_Student.pt"
Parameters_Path = Model_Path[Model_Path.rfind("/")+1: Model_Path.rfind(".")] + "s" # will be "./Parameters/One/Sigma_All_Score_" + str(Parameters_Path) + ".csv"
Data_Path = os.getcwd() + "/../../Datasets/Test_Dataset/"

#Evaluation_Metrics = {"ndcg_1", "ndcg_2", "ndcg_3", "ndcg_4", "ndcg_5", "ndcg_6", "ndcg_7", "ndcg_8", "ndcg_9", "ndcg_10", "ndcg_20"
#                                        , "mrr_1", "mrr_2", "mrr_3", "mrr_4", "mrr_5", "mrr_6", "mrr_7", "mrr_8", "mrr_9", "mrr_10", "mrr_20"}

Evaluation_Metrics = {"ndcg": [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 20], "mrr": [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 20]}
Batch_Size = 1 # 32

#Loss_Name = "listSD"
Loss_Name = "listSDStu"

try:
  Model = torch.load(Model_Path)
except:
  print("Inferenced Model does not exist")
  time.sleep(7)
  exit()

try:
  ### train_ds, Test_Dataset = load_libsvm_dataset(input_path = Data_Path, slate_length = 240, validation_ds_role = "test")
  #
  torch.manual_seed(42)
  torch.cuda.manual_seed_all(42)
  np.random.seed(42)
  Test_Dataset = load_libsvm_dataset_role("test", Data_Path, 240)
  
  #
  """
  print(Test_Dataset.shape)
  if int(np.ceil(Test_Dataset.longest_query_length / 10) * 10) < 240:
    print("----------------------- Loading Again -----------------------")
    torch.manual_seed(42)
    torch.cuda.manual_seed_all(42)
    np.random.seed(42)
    print("slate_length Changed From ", "240", end = " ")
    print("To ", int(np.ceil(Test_Dataset.longest_query_length / 10) * 10))
    Test_Dataset = load_libsvm_dataset_role("test", Data_Path, int(np.ceil(Test_Dataset.longest_query_length / 10) * 10))
    print(Test_Dataset.shape)
  """
  #
except:
  print("There is no data in received Data Path or the format of Data Files is not correct!")
  time.sleep(7)
  exit()

# n_features = train_ds.shape[-1]

#train_dl, Test_DataLoader = create_data_loaders(train_ds, Test_Dataset, num_workers = 1, batch_size = 32)
#

Test_DataLoader = DataLoader(Test_Dataset, batch_size = Batch_Size, num_workers = 0, shuffle=False)


if not os.path.isfile(os.getcwd().replace("\\", "/") + "/Parameters/One/Sigma_All_Score_" + str(Parameters_Path) + ".csv"):
  print("There is no file for Bandwidth Parameters in the Parameters_Path variable")
  print("The Loss Function is automatically change to ListSD")
  print("\nAll The Bandwidth values initialized with 1\n")

  from allrank import config as conf
  #
  Temp = []
  for xb, yb, indices in Test_DataLoader:
    Temp += torch.unique(yb).tolist()
  Unique_Labels = torch.sort(torch.unique(torch.tensor(Temp))).values[1:]
  conf.BandWidth = torch.ones(size = (Unique_Labels.shape[0], xb.shape[-1])).type(torch.float64)
  conf.Best_BandWidth = torch.clone(conf.BandWidth)
  
  Loss_Name = "listSD"




Filtered_Data_Path = ""
for i in Data_Path:
  if i.isalnum():
    Filtered_Data_Path += i


#loss_func = partial(getattr(losses, Loss_Name), **config.loss.args) {"epoch": "a", "Parameters_Path": "b"}
Loss_Function = partial(getattr(losses, Loss_Name), **{"epoch": 1, "Parameters_Path": Parameters_Path}) 


with torch.no_grad():
  Test_losses, Test_nums = zip(
    *[loss_batch(Model, Loss_Function, xb.to(torch.device("cuda")), yb.to(torch.device("cuda")), indices.to(torch.device("cuda")),
         None) for xb, yb, indices in Test_DataLoader])
  #
  Test_metrics = compute_metrics(Filtered_Data_Path, Loss_Name, 1, 1, "test", Evaluation_Metrics, Model, Test_DataLoader, "cuda") # torch.device("cuda")

Test_loss = np.sum(np.multiply(Test_losses, Test_nums)) / np.sum(Test_nums)


print("Loss:", Test_loss)
print(Test_metrics)
