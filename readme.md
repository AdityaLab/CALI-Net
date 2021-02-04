# [AAAI-21] Steering a Historical Disease Forecasting Model Under a Pandemic: Case of Flu and COVID-19

## Publication

Implementation of the paper "Steering a Historical Disease Forecasting Model Under a Pandemic: Case of Flu and COVID-19" published in AAAI-21.

Authors: Alexander Rodríguez, Nikhil Muralidhar, Bijaya Adhikari, Anika Tabassum, Naren Ramakrishnan, B. Aditya Prakash

Venue: AAAI Conference on Artificial Intelligence (AAAI-21)

Pre-print: [https://arxiv.org/abs/2009.11407](https://arxiv.org/abs/2009.11407)

Appendix: [LINK](https://www.cc.gatech.edu/~acastillo41/assets/docs/aaai21-appendix.pdf)

## Requirements

Use the package manager [conda](https://docs.conda.io/en/latest/) to install required Python dependencies. Note: We used Python 3.7.

```bash
conda env create -f requirements.yml
```

## Training

The following command will train and predict for all regions from epidemic week 9 to 15:

```bash
python ./main.py --start_week 9 --end_week 15
```

You can set up your own model hyperparameter values (e.g. learning rate, loss weights) in the file ```./experiment_setup/feature_module/model_specifications/global_recurrent_feature_model.json```.

## Evaluation

To evaluate the results, go to ```evaluate.py``` and change line 71 for the name of results file (saved in folder ```rmse_results```). Then, run.

```bash
python ./evaluate.py
```

## Contact:

If you have any questions about the code, please contact Alexander Rodriguez at arodriguezc[at]gatech[dot]edu and/or B. Aditya Prakash badityap[at]cc[dot]gatech[dot]edu 


