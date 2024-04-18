For Experimental Implementations, the <a href = "https://github.com/allegro/allRank">allrank</a> package of PyTorch is used. Some changes or additive code is utilize on that package which is as following:


The Implementaiotion has 2 phase as the paper says, Teacher Phase and Student Phase respectfully that have ordinal steps below for implementing them. Also, there is a GIF below that show the progress of how to run These Phases in the correct way in details:

### Teacher Phase:
  1. Unpack the <a href = "https://github.com/allegro/allRank">allrank</a> package in the target directory (such as a directory named Teacher).
  2. Download and unpack the SDLR.zip file here (which contains of all the implementation codes in current directory).
  3. Replace or overwrite the unpacked codes of SDLR.zip into the `allrank` directory of the extracted <a href = "https://github.com/allegro/allRank">allrank</a> package.
  4. Set your running settings in the files have names that start with "lambdarank_atmax" of `in` directory inside the `allrank` directory and remember to set the ListSD for name of loss in that setting (Note: the name of loss function is case sensitive).
  5. Go to the main.py in `allrank` directory and in the final lines of the code, change the range of loop with the range of your running settings of `in` directory
     (E.G. if you create and set lambdarank_atmax1.json, lambdarank_atmax2.json up to lambdarank_atmax13.json then you have to set the range of for loop to range(1,14) which will run all those setting files up to lambdarank_atmax13.json).
  7. ss
  
### Student Phase:
  1. Do first three steps of 


### 1. Replacements:
  Replace `main.py` and `config.py` of this directory with same file in `allrank` directory of allrank package.
  
  Replace `train_utils.py` in `training` directory of target package.
  
  Replace `dataset_loading.py` from here within `data` directory of allrank package.
  
  Replace `__init__.py` from here with similar one in `losses` in `models` directory of allrank package.
  
### 2. Adding:
  Add `listSDStu.py` and `listSDStu.py` from here to directory `losses` in `models` directory of allrank packge.
### 3. Changes In Code:
  Change setting of model training at `lambdarank.json` files with the experimental setting in the paper.
  
  Change name of loss to "listSDStu" for Teacher phase and "listSDStus" for Student phase in `lambdarank.json`.
  
  Change the "inupt-norm" value to ${\color{cyan}True}$ for MSLR10K and MSLR30K.
  

<b>Important Note</b>: There is a change in running <i>teacher</i> and <i>student</i>. Two different directory should be made for Student and Teacher with same mentioned changes above. After training of Teacher phase finished, the directory `Parameters` from `allrank` direcotory should copy to `allrank` directory of the Student, then with changing `lambdarank` setting in Student directory, The Student phase training could be started. 

