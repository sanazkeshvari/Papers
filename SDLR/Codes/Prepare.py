import os
import requests
import time

import sys
if len(sys.argv) >= 2:
    dataToTrain = sys.argv[1]
else:
    dataToTrain = "https://raw.githubusercontent.com/sanazkeshvari/Papers/main/SDLR/Code_Help/Test_Dataset.zip"




try:
  import shutil
except:
  """
  import pip
  #pip.main(['help'])
  pip.main(['install', 'pytest-shutil'])
  """
  os.system("py -m pip install pytest-shutil")
  import shutil

"""
import zipfile
with zipfile.ZipFile(path_to_zip_file, 'r') as zip_ref:
    zip_ref.extractall(directory_to_extract_to)
"""



####################### Download the allrank Package #######################
Response = requests.get("https://github.com/allegro/allRank/archive/refs/heads/master.zip")
with open("./allrank.zip", "wb") as F:
  F.write(Response.content)
Response.close()




####################### Download the SDLR Package #######################
#from git import Repo  # pip install gitpython
#Repo.clone_from(git_url, repo_dir)
Response = requests.get("https://raw.githubusercontent.com/sanazkeshvari/Papers/main/SDLR/Codes/SDLR.zip")
with open("./SDLR.zip", "wb") as F:
  F.write(Response.content)
Response.close()




####################### Create Teacher And Student Phases #######################
if not os.path.isdir("./Teacher/"): os.mkdir("./Teacher/")
shutil.unpack_archive("./allrank.zip", "./Teacher/")


## Add SDLR to allrank Packege 
shutil.unpack_archive("./SDLR.zip", "./Teacher/allRank-master/allrank/") 


if os.path.isdir("./Student/"): shutil.rmtree("./Student/")
shutil.copytree("./Teacher/allRank-master/", "./Student/")
#shutil.copytree("./Teacher/allRank-master/", "./Teacher/") # , dirs_exist_ok = True, ignore = True
shutil.rmtree("./Teacher/")
shutil.copytree("./Student/", "./Teacher/", )




####################### Start Checking By A Sample Dataset #######################
print("\nIn the following, The code will be trained over the provided train data. The defult trian sample data is availabe in code_help")
print("\n####################### #######################\n")
print("You can pause training")
print("\t By closing the running window or press Ctrl + C keys")
print("\n####################### #######################")
for i in range(9):
  print(9 - i, end = "\t")
  time.sleep(0.99)


####################### Download Sample Data (Test_Dataset) #######################
print("\n\nDownloading  Dataset...")
Response = requests.get(dataToTrain)
with open("./Test_Dataset.zip", "wb") as F:
  F.write(Response.content)
Response.close()

#if not os.path.isdir("./Data/MQ2008/Fold1/"): os.makedirs("./Data/MQ2008/Fold1/")
shutil.unpack_archive("./Test_Dataset.zip", "./Datasets/Test_Dataset/")



####################### Determining The Run Setting #######################
Info_File = open("./Datasets/Test_Dataset/lambdarank_atmax1.json", "r")
Info = Info_File.read()
Info_File.close()

Dataset_Address_Index = Info.find("path_to_dataset")
Info = Info.replace(Info[Dataset_Address_Index - 1: Dataset_Address_Index + 16], os.getcwd().replace("\\", "/") + "/Datasets/Test_Dataset/")

Info_File = open("./Datasets/Test_Dataset/lambdarank_atmax1.json", "w")
Info_File.write(Info)
Info_File.close()

shutil.copy("./Datasets/Test_Dataset/lambdarank_atmax1.json", "./Teacher/allrank/in/lambdarank_atmax1.json")



Info_File = open("./Datasets/Test_Dataset/lambdarank_atmax1.json", "r")
Info = Info_File.read()
Info_File.close()

Info = Info.replace("listSD", "listSDStu")

Info_File = open("./Datasets/Test_Dataset/lambdarank_atmax1.json", "w")
Info_File.write(Info)
Info_File.close()

shutil.move("./Datasets/Test_Dataset/lambdarank_atmax1.json", "./Student/allrank/in/lambdarank_atmax1.json")




####################### Checking Requirements #######################
"""
try:
  import numpy as np
  import pandas as pd
  import torch
  import urllib
  import gcsfs
  import tensorboardX
except:
  #os.system("py -m pip install numpy pandas torch urllib gcsfs tensorboardX")
  os.system("py -m pip install -r ./requirements.txt")
  import numpy as np
  import pandas as pd
  import torch
  import urllib
  import gcsfs
  import tensorboardX
"""
os.system("py -m pip install -r ./requirements.txt")



print("\n\n -----------------------\n Teacher Phase \n -----------------------\n\n")

os.system("cd Teacher/allrank/ && py main.py")

try:
  shutil.copytree("./Teacher/allrank/Parameters/", "./Student/allrank/Parameters/")
  print("Bandwidths values that were computed by the Teacher is copied to the Student Phase")


  print("\n\n -----------------------\n Student Phase \n -----------------------\n\n")

  os.system("cd Student/allrank/ && py main.py")

except:
  print("Your device does not meet the requirements.\n")

print("The results of the Teacher Phase and the Student Phase is accessible in -allrank- directory of each one in Sheet Files which ends with the name of given Dataset")
print()
