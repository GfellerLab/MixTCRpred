# TCRpred
TCRpred is a sequence-based TCR-epitope interaction predictor. TCR binding predictions are currently possible for 146 pMHCs.
The paper describing the TCR-pMHC dataset, the architecture of the TCRpred as well as the predictive performances and potential application is available at [here](XXXX)

## Installation

1. Clone the GitHub repository and move to the TCRpred directory
```bash
git clone https://github.com/GfellerLab/TCRpred 
mv TCRpred
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
``
which will show all the available TCRpred models.

## Usage

```bash
(TCRpred_venv) python3 TCRpred.py --epitope [TCRpred_model_name] --input [input_TCR_file] --out [output_file]
```

For example to test which TCRs of the file ./test/test.out can bind to the HLA-A\*02:01, GILGFVFTL run 

```bash
(TCRpred_venv) python3 TCRpred.py --epitope A0201_ELAGIGILTV --input ./test/test.csv --out ./test/out.csv
```


In the GitHub repository we include only two TCRpred models (A0201_GILGFVFT and A0201_ELAGIGILTV). 

To download all the 146 pretrained TCRpred models from (Zenodo) run:
```bash
bash code to download from zenodo
```
or download TCR_pretrained_models.zip from (zenodo link), unzip it and replace the pretrained_model folders

The code was tested with python 3.9

