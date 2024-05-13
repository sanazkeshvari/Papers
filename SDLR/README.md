# A Self-Distilled Learning to Rank Model for Ad-hoc Retrieval

 This is the GitHub repository for the Self-Distilled Learning to Rank (SDLR) framework. SDLR is a novel approach introduced in "A Self-Distilled Learning to Rank Model for Ad-hoc Retrieval". This paper proposes an innovative framework for ad-hoc retrieval, aimed at improving retrieval performance while robustly handling noisy and outlier data during the training process.  SDLR assigns a confidence weight to each training sample, aiming at reducing the impact of noisy and outlier data in the training process. The confidence wight is approximated based on the feature’s distributions derived from the values observed for the features of the documents labeled for a query in a listwise training sample. SDLR includes a distillation process that facilitates passing on the underlying patterns in assigning confidence weights from the teacher model to the student one.
<div class="row">
   <center>
 <img class="img-responsive center-block" width="600" alt="model1" src="https://github.com/sanazkeshvari/Papers/blob/main/SDLR/newDesign.png">
   </center>
</div>


## Code

All the required code for training both teacher and student models can be found in /Codes/

All required packages are listed in `/Codes/requirements.txt`

Some of the important packages are as follows:

1. numpy
2. pandas
3. pytorch (known as torch)
4. shutil (pip install pytest-shutil)
6. gcsfs
7. tensorboardX

 <b> Important Note:</b> This code needs a system with <b>GPU</b>. Also, ensure that your torch version is compatible with <b>CUDA</b>.

(Note that the basic Python package is not listed here and is assumed to be installed.)

<hr/>

In our codes, we use and modify the learning to rank modules from  <a href = "https://github.com/allegro/allRank">allrank</a>.

In the following, we explain two different ways you can train the teacher and student models of the SDLR framework. First, a Python script, which downloads, extracts, and runs all the necessary code. Second, there is a detailed explanation of how to manually download and run the code.


### First: Running the script
1. Download <a href = "https://raw.githubusercontent.com/sanazkeshvari/Papers/main/SDLR/Codes/Prepare.py" target = "_blank" download="SDLR">Prepare.py</a>
2. Place Prepare.py code in the destination directory.
3. Prepare.py gets the web address of the dataset (The data on which models can be trained) as the input. The defualt value for this input is the sample database we provided in  <a href = "https://github.com/sanazkeshvari/Papers/tree/main/SDLR/Code_Help/Test_Dataset.zip"> Code_Help/Test_Dataset.zip </a>
4. The expected format of the dataset is a zip file consisting of the following files: train, test, dev, and setting. The setting file has been placed separately in the 'Code_Help' directory ('Code_Help/setting.json'). Please note that this is only one setting among all the settings tested and reported in the paper. You can manually change the setting file.
5. After installing all the required packages, use Python 3 for running Preparer.py.

### Second: Manual Preparation of codes and data

Training this framework has two phases, as described in the paper: the Teacher Phase and the Student Phase. Each phase consists of the follwoing steps. Additionally, there are two videos that show the detailed process of training these phases correctly.

#### Teacher Phase:

  1. Unpack the <a href = "https://github.com/allegro/allRank">allrank</a> package in a target directory (such as a directory named Teacher).
  2. Download and unpack the SDLR.zip file (which contains all the implementation codes).
  3. Replace or add the unpacked codes of SDLR.zip into the `allrank` directory of the extracted <a href = "https://github.com/allegro/allRank">allrank</a> package (See Replacements section in this manual).
  4. Set your running settings in the files whose names start with "lambdarank_atmax" of `in` directory inside the `allrank` directory and remember to set the datasets address in "path" of "data" and "ListSD" for name of loss in that setting (Note: the name of loss function is case sensitive).

     (Change the "inupt-norm" of setting file to True for MSLR10K and MSLR30K.) 
     (Datasets address should be a directory address that contains of train.txt, vali.txt, and test.txt and it should be mentioned that which of vali.txt and test.txt should be used for validation data of training)
  6. Go to the "main.py" in `allrank` directory and in the final lines of the code, change the range of loop with the range of your running settings of `in` directory then run the "main.py".

     (E.G. if you create and set lambdarank_atmax1.json, lambdarank_atmax2.json up to lambdarank_atmax13.json then you have to set the range of for loop to range(1,14) which will run all those setting files up to lambdarank_atmax13.json).
  7. Results will be store into `allrank` directory in csv files with name ends with the "ListSD.csv" and Bandwidths values will be saved in directory named `One` into the `Parameters` directory which will be created in `allrank` directory after running "main.py".


See Steps of Teacher Phase in This <a href = "https://github.com/sanazkeshvari/Papers/blob/06bf8bf07bc461a035cabb797ecd50bd24b66b7a/SDLR/Code_Help/SDLR_Teacher_20240418_VeryFast1080.mp4">Video</a>.


  
#### Student Phase:
  1. Go through the first three steps of the Teacher Phase sequentially.
  2. Copy the Parameters directory from the Teacher Phase's allrank directory (which includes saved Bandwidth values obtained during the Teacher Phase) into the allrank directory of the Student Phase.
  3. Configure your running settings as in step 4 of the Teacher Phase, but ensure that the loss function for the Student Phase is set to "ListSDStu" (Note: the name of the loss function is case-sensitive).

     (Change the "inupt-norm" of setting file to True for MSLR10K and MSLR30K.).
  4. Execute the "main.py" script to train the Student Phase, following step 6 of training the Teacher model.
  5. The results will be stored in the allrank directory in CSV files with names ending in "ListSDStu.csv".

See Steps of Student Phase in This <a href = "https://github.com/sanazkeshvari/Papers/blob/06bf8bf07bc461a035cabb797ecd50bd24b66b7a/SDLR/Code_Help/SDLR_Student_20240418_VeryFast1080.mp4">Video</a>.


#### Additional Notes:
  1. For both Student Phase and Teacher Phase, some directories whosw names start with "out" will be created next to `allrank` directory which contains of the logs ans information of running the code for the defined setting (Such as `out2` directory for running "lambdarank_atmax2.json").
  2.For the <b>Robustness</b> experiment, there are two different types of noises as described in the experimental section of the paper: feature-based noise and normal distribution noise with a defined variance. To address this, two parameters have been added to the code, which are accessible in the last lines of the 'main.py' file: <br/> <br/>
 The first parameter is "<b>Noise_Percent</b>," which specifies the portion of data randomly chosen to have noise added to it. This value can range between 0.0 and 1.0, representing 0% and 100% of the data, respectively. <br/> <br/>
The second parameter is "<b>Max_Noise</b>," which determines the type of noise with three different options:

a. If it is equal to 0, it means there is no noise.

b. If its value is positive, it will add Normal Distribution Noise with the specified value. This means that the values of the additive noise will range from negative to positive of that value. For example, if Max_Noise equals 0.05, it means the values of Normal Distribution Noise will be in the range of -0.05 to 0.05.

c. If the value of Max_Noise is set to a negative value, the noise will be added to the data based on the variance of each feature.

  4.For changing the amount of data used for training in both the Teacher Phase and the Student Phase, the "Data_Percent" parameter is added to the code in the last lines of main.py. The value of "Data_Percent" can be adjusted between 0.0 and 1.0, representing 0% and 100% of the data, respectively."
  
  6. At the end of running all implementations for all settings in the main.py (after completing all training sessions), a CSV file named "Run_Times.csv" will be created in the allrank directory. This file shows the runtime of the training for each training setting.


<br/> <br/> <br/> <br/> <br/>


### Replacements:
1. Replacements: main.py, and config.py,`train_utils.py`, `dataset_loading.py` , `__init__.py`.
  
2. Adding: two new losses have been added: `listSDStu.py` and `listSDStu.py` in the  `losses` directory.






