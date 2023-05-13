# TCRpred
TCRpred is a sequence-based TCR-pMHC interaction predictor. 
TCR binding predictions are currently possible for 146 pMHCs. 
For 43 pMHC robust predictions were achieved in internal cross validation, while models with less than 50 training TCRs have low confidence.
[Here](XXXX) the paper describing TCRpred predictive performance and applications.


## Run TCR predictions in your browser

By accessing [this link](https://colab.research.google.com/github/GiancarloCroce/test/blob/main/colab_TCRpred.ipynb), 
TCRpred can be directly used in your web browser through Google Colab, providing an intuitive and interactive method 
to test your own TCR list against any of the 146 pMHC TCRpred models. 
For more extensive analysis or if you wish to use TCRpred offline, follow the installation instructions.

## Install TCRpred on your local machine

Alternatively, you can download TCRpred and run it in your machine. The code was tested with python 3.9

1. Clone the GitHub repository and move to the TCRpred directory
```bash
git clone https://github.com/GfellerLab/TCRpred 
cd TCRpred
```

2. (Recommended) Create a virtual environment to install the required packages
```bash
python3 -m venv TCRpred_venv  
source TCRpred_venv/bin/activate  # to activate the virtual environment
(TCRpred_venv) pip install -r requirements.txt  # to install the packages
```

3. To test your installation, run the following command:
```bash
(TCRpred_venv) python TCRpred.py --help
```

which will show all the available TCRpred models.
![](help_output.png)  

## Usage

```bash
(TCRpred_venv) python3 TCRpred.py --model [TCRpred_model_name] --input [input_TCR_file] --out [output_file]
```

For example to test which TCRs of the file ./test/test.out can bind to the HLA-A\*02:01, GILGFVFTL run 
```bash
(TCRpred_venv) python3 TCRpred.py --model A0201_GILGFVFTL --input ./test/test.csv --out ./test/out.csv
```


Three arguments are required:

--model [TCRpred_model_name]
TCRpred model. To get the list of available models use --help. 
The format is HLAname_PeptideSequence (A0201_GILGFVFTL).

--input [input_TCR_file]:
csv file listing all the TCRs to test. See ./test/test.cvs for a reference input file. The columns order is not important.
CDR3 alpha and beta should not be longer than 20 amino acids.
Incomplete TCR entries are accepted, but models have lower predictive performance

--output [output_file]
The name of the output file. It contains two more column than the input: the TCRpred binding score and the %rank.

(Optional) --bath_size [batch_size]
The default batch size is 1. If you have a large dataset of TCRs to test, increasing the batch_size can speed TCRpred up.

## Download TCRpred pretrained models

In the GitHub repository we include only two TCRpred models (A0201_GILGFVFT and A0201_ELAGIGILTV). 
You can download the pretrained TCRpred models from our [Zenodo dataset](https://doi.org/10.5281/zenodo.7930623)

To download all the 146 pretrained TCRpred models run:
```bash
(TCRpred_venv) python3 TCRpred.py --download
```

To download the high-confidence 43 models (more than 50 training TCRs) run:
```bash
(TCRpred_venv) python3 TCRpred.py --download_high
```
