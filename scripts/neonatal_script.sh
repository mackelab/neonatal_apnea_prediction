#!/bin/bash
cd ../src

main_id=$1

echo START MAIN EXPERIMENT
python3 train_neonatal.py meta.experiment=${main_id}_main_multirun meta.tag=${main_id}_run_01


echo START SINGLE MODALITY EXPERIMENT
python3 train_neonatal.py meta.experiment=${main_id}_np_multirun +experiment=single_mod_np meta.tag=${main_id}_np_01
python3 train_neonatal.py meta.experiment=${main_id}_th_multirun +experiment=single_mod_th meta.tag=${main_id}_th_01
python3 train_neonatal.py meta.experiment=${main_id}_hr_multirun +experiment=single_mod_hr meta.tag=${main_id}_hr_01
python3 train_neonatal.py meta.experiment=${main_id}_pr_multirun +experiment=single_mod_pr meta.tag=${main_id}_pr_01
python3 train_neonatal.py meta.experiment=${main_id}_spo2_multirun +experiment=single_mod_spo2 meta.tag=${main_id}_spo2_01
python3 train_neonatal.py meta.experiment=${main_id}_pco2_multirun +experiment=single_mod_pco2 meta.tag=${main_id}_pco2_01


echo START LAG EXPERIMENT
python3 train_neonatal.py meta.experiment=${main_id}_lag_experiment dataset.lag=0 meta.tag=${main_id}_lag0
python3 train_neonatal.py meta.experiment=${main_id}_lag_experiment dataset.lag=3000 meta.tag=${main_id}_lag15
python3 train_neonatal.py meta.experiment=${main_id}_lag_experiment dataset.lag=6000 meta.tag=${main_id}_lag30
python3 train_neonatal.py meta.experiment=${main_id}_lag_experiment dataset.lag=9000 meta.tag=${main_id}_lag45
python3 train_neonatal.py meta.experiment=${main_id}_lag_experiment dataset.lag=12000 meta.tag=${main_id}_lag60
python3 train_neonatal.py meta.experiment=${main_id}_lag_experiment dataset.lag=15000 meta.tag=${main_id}_lag75
python3 train_neonatal.py meta.experiment=${main_id}_lag_experiment dataset.lag=18000 meta.tag=${main_id}_lag90
python3 train_neonatal.py meta.experiment=${main_id}_lag_experiment dataset.lag=21000 meta.tag=${main_id}_lag105
python3 train_neonatal.py meta.experiment=${main_id}_lag_experiment dataset.lag=24000 meta.tag=${main_id}_lag120
python3 train_neonatal.py meta.experiment=${main_id}_lag_experiment dataset.lag=27000 meta.tag=${main_id}_lag135
python3 train_neonatal.py meta.experiment=${main_id}_lag_experiment dataset.lag=30000 meta.tag=${main_id}_lag150