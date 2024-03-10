# DI-OO
The source code for paper: An  Effective Data Imputation Order Optimization Framework with Meta-Learning and Multi-Armed Bandits
1. building Constructing datasets for an incomplete dataset by using build_space.py 
       example : build candidate datasets for "Supreme" , which systematically lacks 30% of its data cells

   ​	`python build_space.py --dataset "Supreme" --mv_prob 0.3 --mv_type systematic`

2. Using the DI-OO system to impute the incomplete dataset based on the downstream model

   We provide examples of two downstream models, i.e. DI-OO_CNN.py and DI-OO_MLP.py
