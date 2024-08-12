# DI-OO
The source code for paper: Think Twice Before Imputation: Optimizing Data Imputation Order for Machine Learning
1. building Constructing datasets for an incomplete dataset by using build_space.py 
       example : build candidate datasets for "Supreme" , which systematically lacks 30% of its data cells

   ​	`python build_space.py --dataset "Supreme" --mv_prob 0.3 --mv_type systematic`

2. Using the DI-OO system to impute the incomplete dataset based on the downstream model

   We provide examples of two downstream models, i.e. DI-OO_CNN.py and DI-OO_MLP.py
   
   `python DI-OO_MLP.py --dataset "cancellation" --data_size 2000 --value 0.1 --input_size 17 --param_sens 0.1 --mv_type systematic --n_arms 3`

# Reference
https://github.com/chu-data-lab/CPClean
