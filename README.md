
The proposed cross-domain task coordinator (CDTC) for cross-domain few-shot graph learning. 


## Environment  
We used the following Python packages for core development. We tested on `Python 3.7`.
```
- pytorch 1.7.0
- torch-geometric  2.0.3
- torch-cluster             1.5.8                   
- torch-scatter             2.0.5                  
- torch-sparse              0.6.8                  
- torch-spline-conv         1.2.0
```

## Datasets 
Tox21, SIDER, MUV are public datasets from molecular property prediction benchmarks from [SNAP](http://snap.stanford.edu/gnn-pretrain/data/chem_dataset.zip). 

## Experiments
Example of executing the cross-domain experiments:
```
python main_ms2t_rl.py --test_dataset $TESTDATA$ --d_names $TRAINDATA-TESTDATA$ --eval_steps 10 --inner_lr $INNER_LR$ --update_step_test  4  --batch_task 4 --seed 5  --naivedim $SOURCE_TASK_NUMBER$  --meta_lr 0.0005  --n_shot_train 10 --n_shot_test 10 --pretrained $PRETRAINED_OR_NOT$ --epochs 2000 --enc_gnn $GNN_TYPE$ --maml --applyrl  --rl_adv $AGENT_LR2$  --step_gin1 10 --step_agent 500
```
