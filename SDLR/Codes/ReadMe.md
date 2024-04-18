For Experimental Implementations, the <a href = "https://github.com/allegro/allRank">allrank</a> package of PyTorch is used. Some changes or additive code is utilize on that package which is as following:


The Implementaiotion has 2 phases as the paper says, Teacher Phase and Student Phase respectfully that each has ordinal steps below for implementing them. Also, there are two GIF below that show the progress of how to run These Phases in the correct way in details:

### Teacher Phase:
  1. Unpack the <a href = "https://github.com/allegro/allRank">allrank</a> package in the target directory (such as a directory named Teacher).
  2. Download and unpack the SDLR.zip file here (which contains of all the implementation codes in current directory).
  3. Replace or overwrite the unpacked codes of SDLR.zip into the `allrank` directory of the extracted <a href = "https://github.com/allegro/allRank">allrank</a> package.
  4. Set your running settings in the files have names that start with "lambdarank_atmax" of `in` directory inside the `allrank` directory and remember to set the "ListSD" for name of loss in that setting (Note: the name of loss function is case sensitive).

     (Change the "inupt-norm" of setting file to ${\color{cyan}True}$ for MSLR10K and MSLR30K.)
  6. Go to the "main.py" in `allrank` directory and in the final lines of the code, change the range of loop with the range of your running settings of `in` directory then run the "main.py".

     (E.G. if you create and set lambdarank_atmax1.json, lambdarank_atmax2.json up to lambdarank_atmax13.json then you have to set the range of for loop to range(1,14) which will run all those setting files up to lambdarank_atmax13.json).
  7. Results will be store into `allrank` directory in csv files with name ends with the "ListSD.csv" and Bandwidths values will be saved in directory named `One` into the `Parameters` directory which will be created in `allrank` directory after running "main.py".
  
### Student Phase:
  1. Go through first three steps of Teacher Phase ordinally.
  2. Copy The `Parameters` directory of `allrank` directory from Teacher Phase (which includes of saved Bandwidths value that has reached by Teacher Phase) into the `allrank` directory of Student Phase.
  3. Set you running setting as the step 4 of Teacher Phase But loss function for Student Phase should be "ListSDStu"(Note: the name of loss function is case sensitive).

     (Change the "inupt-norm" of setting file to ${\color{cyan}True}$ for MSLR10K and MSLR30K.).
  4. Run the "main.py" for training the Student Phase as the step 6 of Teacher Phase.
  5. Results will be store into `allrank` directory in csv files with name ends with the "ListSDStu.csv"


<br/> <br/> <br/>

### Additional Notes:
  1. For both Student Phase and Teacher Phase, some directories with names start with "out" will be created next to `allrank` directory which contains of the logs ans information of running the code for the defined setting (Such as `out2` directory for running "lambdarank_atmax2.json").
  2. For <b>Robustness</b> experimental facing with the <b>Normal Distribution Noise</b>, there is 2 different type as the experimental section of the paper, feature based noise and normal distribution noise with defined variance. For this target, There are two parameters added to the codes which is accessible in the last lines of the "main.py": <br/> <br/>
  The first Parameter is "<b>Noise_Percent</b>" which says the portion of data which will be choose randomly for adding noise to them that could be change between 0.0 and 1.0 those for 0% and 100% of data respectfully. <br/> <br/>  
  The second parameter is "<b>Max_Noise</b>" that determine the type of noise in 3 different options: <br/> 
     a. if it is equal to 0 means there is no noise or the results is for Not Noisy data. <br/>
        
     b. if its value is positive it will add a Normal Distribution Noise with the set value that means the values of additive noise are in the range of minus and positive of that value (E.G. if Max_Noise equals 0.05 then it mean the values of Normal Distribution Noise will be in the range of -0.05 and 0.05). <br/>   
     c. if the value of Max_Noise was set to a minus value then the noise will be add to data based on the variance of each feature. <br/>
  3. For changing the amount of data for training in Teacher Phase and Student phase, the "Data_Percent" parameters added to the code in the last lines of main.py. The value of "Data_Percent" could be change between 0.0 and 1.0 which is for 0% and 100% of data respectfully.



<br/> <br/> <br/> <br/> <br/>

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

