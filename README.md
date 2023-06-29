# TCRpred
TCRpred is a sequence-based TCR-pMHC interaction predictor. 
TCR binding predictions are currently possible for 146 pMHCs. 
For 43 pMHC robust predictions were achieved in internal cross validation, while models with less than 50 training TCRs have low confidence.
[Here](XXXX) the paper describing TCRpred predictive performance and applications.


## Run TCRpred with GoogleColab
You can run TCRpred within your web browser via Google Colab by clicking on [this link](https://colab.research.google.com/github/GfellerLab/TCRpred/blob/main/colab_TCRpred.ipynb).
This is a user-friendly and interactive way to analyze the specificity of your own TCR list.
For more extensive analysis or if you prefer to use TCRpred offline, it is recommended to install it on your local machine.

## Install TCRpred 

1. Clone the GitHub repository and move to the TCRpred directory
```bash
git clone https://github.com/GfellerLab/TCRpred 
cd TCRpred
```

2. (Recommended) Create a virtual environment and install the required packages
```bash
# For Unix/Mac OS users
python -m venv TCRpred_venv  
source TCRpred_venv/bin/activate  # to activate the virtual environment (TCRpred_venv)
pip install --upgrade pip
pip install -r requirements.txt  # to install the packages

# Windows users please refer to https://docs.python.org/3/library/venv.html to create and activate a virtual environment. 

```

3. To test your installation, run the following command:
```bash
python TCRpred.py --help
```
![](help_output.png)  

or 

```bash
python TCRpred.py --list_model
```

which will show all the available TCRpred models.
![](list_model_output.png)

4. Finally run 

```bash
python TCRpred.py --model A0201_GILGFVFTL --input ./test/test.csv --output ./test/output.csv 
```

to predict which TCRs in the ./test/test.csv file are more likely to binding to the HLA-A\*02:01,GILGFVFTL epitope.



5. (Optional) To run TCRpred from anywhere on the computer, open TCRpred.py with your favourite editor and specify the full path to the pretrained models folder:
```bash
#change
path_pretrained_models = './pretrained_models'
#to 
path_pretrained_models = '/home/[...]/TCRpred/pretrained_models'
```
Next make an alias to the TCRpred.py file using the python version of the virtual enviroment:
```bash
# For Unix/Mac OS users
alias TCRpred='/home/[...]/TCRpred/TCRpred_venv/bin/python /home/[...]/TCRpred/TCRpred.py'
# you can make this alias permanent by adding it to your .bashrc file
```

## Usage

```bash
source TCRpred_venv/bin/activate  # to activate the virtual environment (TCRpred_venv)
python TCRpred.py --model [TCRpred_model_name] --input [input_TCR_file] --out [output_file]
```

Three arguments are required:

``` --model``` or  ``` -m``` [TCRpred_model_name]. 
To get the list of available models use ``` --help ```. 
The format is HLAname_PeptideSequence (e.g. A0201_GILGFVFTL).

```--input``` or ```-i``` [input_TCR_file].
csv file listing all the TCRs to test. See ./test/test.cvs for a reference input file. The columns order is not important.
CDR3 alpha and beta should not be longer than 20 amino acids.
Incomplete TCR entries are accepted, but models have lower predictive performance

``` --output``` or ```-o``` [output_file].
The name of the output file. It contains two more column than the input: the TCRpred binding score and the %rank.


Additional and optional arguments are:  
```--list_model```. To visualize the 146 TCRpred models for which we can run predictions. Models with less than 50 training TCRs have low confidence  
```--batch_size```. The default batch size is 1. If you have a large dataset of TCRs to test, increasing the batch_size can speed TCRpred up  
```--download model_name```. To download a specific pretrained TCRpred model  
```--download_all```. To download the 146 pretrained TCRpred models  
```--download_high```. To download the 43 high-confidence pretrained TCRpred models 
```-h``` or ```--help```. To print the help message.


## Download TCRpred pretrained models

In the GitHub repository we include only two TCRpred models (A0201_GILGFVFT and A0201_ELAGIGILTV). 
You can download the pretrained TCRpred models from our [Zenodo dataset](https://doi.org/10.5281/zenodo.7930623)

To download a specific pretrained model (e.g. A0201_NLVPMVATV) run:
```bash
python TCRpred.py --download A0201_NLVPMVATV 
```

To download all the 146 pretrained TCRpred models run:
```bash
python TCRpred.py --download
```

To download the high-confidence 43 models (more than 50 training TCRs) run:
```bash
python TCRpred.py --download_high
```

## Contact information

For scientific questions, please contact [Giancarlo Croce](mailto:giancarlo.croce@unil.ch?subject=[GitHub]%20TCRpred%20) or [David Gfeller](mailto:david.gfeller@unil.ch?subject=[GitHub]%20TCRpred%20)

For license-related questions, please contact [Nadette Bulgin](mailto:nbulgin@lcr.org?subject=[GitHub]%20TCRpred%20).
