# TCRpred
TCRpred is a sequence-based TCR-epitope interaction predictor. TCR binding predictions are currently possible for 146 pMHCs.
The paper describing the TCR-pMHC dataset, the architecture of the TCRpred as well as the predictive performances and potential application is available at [here](XXXX)

## Installation

1. Clone the GitHub repository and move to the TCRpred directory
```bash
git clone https://github.com/GfellerLab/TCRpred 
cd TCRpred
```

2. (Recommended) Create a virtual environment to install the required packages
```bash
python3 -m venv TCRpred_venv  
source TCRpred_venv/bin/activate  # to activate the virtual environment
(TCRpred_venv) pip3 install -r requirements.txt  # to install the packages
```

3. To test your installation, run the following command:
```bash
(TCRpred_venv) python3 TCRpred.py --help
```

which will show all the available TCRpred models.

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


In the GitHub repository we include only two TCRpred models (A0201_GILGFVFT and A0201_ELAGIGILTV). 

To download all the 146 pretrained TCRpred models from (Zenodo) run:
```bash
bash code to download from zenodo
```
or download TCR_pretrained_models.zip from (zenodo link), unzip it and replace the pretrained_model folders

The code was tested with python 3.9

