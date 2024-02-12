For Experimental Implementations, the <a href = "https://github.com/allegro/allRank">allrank</a> package of Tensorflow is used. Some changes or additive code is utilize on that package which is as following:

### 1. Replacements:
  Replace `main.py` of this directory with same file in `allrank` directory of allrank package.
  
  Replace `train_utils.py` in `training` directory of target package.
  
  Replace `dataset_loading.py` from here within `data` directory of allrank package.
  
  Replace `__init__.py` from here with similar one in `losses` in `models` directory of allrank package.
  
  Replace `config.py` with existed file in ` ` directory.
### 2. Adding:
  Add `listSDStu.py` and `listSDStu.py` from here to directory `losses` in `models` directory of allrank packge.
### 3. Changes In Code:
  Change setting of model training at `lambdarank.json` files with the experimental setting in the paper.
  
  Change name of loss to "listSDStu" for Teacher phase and "listSDStus" for Student phase in `lambdarank.json`.
  
  Change the "inupt-norm" value to ${\color{cyan}True}$ for MSLR10K and MSLR30K.
  

Important Note: There is a change in running teacher and student. Two different directory should be made for Student and Teacher with same mentioned changes above. After training of Teacher phase finished, the directory `Parameters` from `allrank` direcotory should copy to `allrank` directory of the Student, then with changing `lambdarank` setting in Student directory, The Student phase training could be started. 


"Compute_BandWidth_torch" and "Train_utils"  in training folder

