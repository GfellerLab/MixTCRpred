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
python -m venv TCRpred_venv  
source TCRpred_venv/bin/activate  # to activate the virtual environment (TCRpred_venv)
pip install --upgrade pip
pip install -r requirements.txt  # to install the packages
```

3. To test your installation, run the following command:
```bash
python TCRpred.py --help
```

which will show all the available TCRpred models.
![](help_output.png)  


4. (Optional) To run TCRpred from anywhere on the computer, open TCRpred.py with your favourite editor and specify the full path to the pretrained models folder:
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
To test your installation, run the following command:
```bash
TCRpred --help
```

## Usage

```bash
source TCRpred_venv/bin/activate  # to activate the virtual environment (TCRpred_venv)
python TCRpred.py --model [TCRpred_model_name] --input [input_TCR_file] --out [output_file]
```

For example to test which TCRs of the file ./test/test.out are likely to interact with HLA-A\*02:01, GILGFVFTL, you can run 
```bash
python TCRpred.py --model A0201_GILGFVFTL --input ./test/test.csv --out ./test/out.csv
```


Three arguments are required:

``` --model ``` or  ``` -m ``` [TCRpred_model_name]. 
To get the list of available models use ``` --help ```. 
The format is HLAname_PeptideSequence (A0201_GILGFVFTL).

```--input ``` or ``` -i ``` [input_TCR_file].
csv file listing all the TCRs to test. See ./test/test.cvs for a reference input file. The columns order is not important.
CDR3 alpha and beta should not be longer than 20 amino acids.
Incomplete TCR entries are accepted, but models have lower predictive performance

``` --output ``` or ``` -o ``` [output_file].
The name of the output file. It contains two more column than the input: the TCRpred binding score and the %rank.

(Optional) ``` --bath_size ``` [batch_size].
The default batch size is 1. If you have a large dataset of TCRs to test, increasing the batch_size can speed TCRpred up.

## Download TCRpred pretrained models

In the GitHub repository we include only two TCRpred models (A0201_GILGFVFT and A0201_ELAGIGILTV). 
You can download the pretrained TCRpred models from our [Zenodo dataset](https://doi.org/10.5281/zenodo.7930623)

To download all the 146 pretrained TCRpred models run:
```bash
python TCRpred.py --download
```

To download the high-confidence 43 models (more than 50 training TCRs) run:
```bash
python TCRpred.py --download_high
```
