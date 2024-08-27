# MixTCRpred (version for NY-ESO-1)
MixTCRpred is a predictor of interaction between T-cell receptors (TCRs) and viral and cancer epitopes (peptides displayed on MHC molecules or pMHCs). Predictions are available for 146 pMHCs. [Here](https://www.nature.com/articles/s41467-024-47461-8) the paper describing MixTCRpred predictive performance and applications. Accurate predictions were achieved for 43 pMHCs that have more than 50 training TCRs. 

This branch is specifically for the NY-ESO-1<sub>157-165</sub> epitope. As described [in this paper](xxx), we used a phage display screening to screen thousands of TCRs with randomized CDR3b against the NY-ESO-1 epitope. We used such data to train a MixTCRpred model for this epitope.  The MixTCRpred model name is "A0201_NY-ESO-1-CDR3b". 


### Usage for A0201_NY-ESO-1-CDR3b

For this MixTCRpred model:

- **Only TCRs with TRBV6-5, TRBJ2-2, beginning with "CASS" and ending with "GELFF" in the CDR3 beta will be accepted as inputs. All other sequences will be discarded.**
- The alpha chain sequences are not used for predicting the TCR specificity. TCRs with alpha chain sequences equal to the reference TCR used in the phage display screening (TRAV21 | CAVRPTSGGSYIPTF | TRAJ6) are accepted as inputs, while other alpha chains will be replaced by null values.


To utilize the "A0201_NY-ESO-1-CDR3b" MixTCRmodel, execute the following command:

```bash
python MixTCRpred.py --model A0201_NY-ESO-1-CDR3b --input [input_TCR_file] --output [output_file]
```

Please refer to the license LICENCE_A0201_NY-ESO-1-CDR3b.md in order to use "A0201_NY-ESO-1-CDR3b" MixTCRpred model.



## Contact information

For scientific questions, please contact [Giancarlo Croce](mailto:giancarlo.croce@unil.ch?subject=[GitHub]%20MixTCRpred%20) or [David Gfeller](mailto:david.gfeller@unil.ch?subject=[GitHub]%20MixTCRpred%20)

For license-related questions, please contact [Nadette Bulgin](mailto:nbulgin@lcr.org?subject=[GitHub]%20MixTCRpred%20).

## Acknowledgments

This project received funding from the European Union's Horizon 2021 research and innovation programme under the Marie Skłodowska-Curie grant agreement, No. 101027973, [MT-PoINT project](https://cordis.europa.eu/project/id/101027973)
