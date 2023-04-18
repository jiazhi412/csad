# given pretrain
python main.py -e csad0020 --color_var 0.02 --checkpoint baseline/pretraincheckpoint_step_0000.pth --alpha 1 --tau 10 --lambda_ 1 
python main.py -e debug --color_var 0.05 

# own train
python main.py -e append_test --color_var 0.02 
python main.py -e append_test --color_var 0.02 

# CMNIST
python main.py -e CMNIST_debug_1003

# CelebA
python main_CelebA.py -e CelebA_debug_1003

# IMDB 
python main_IMDB.py -e IMDB_debug_1003 --IMDB_train_mode eb1 --IMDB_test_mode eb2
python main_IMDB.py -e IMDB_debug_1003 --IMDB_train_mode eb1_ex --IMDB_test_mode eb2_ex

# Adult 
python main_Adult.py -e Adult_debug_117 --Adult_train_mode eb1 --Adult_test_mode eb2

# German
python main_German.py -e German_debug_130 --German_train_mode eb1 --German_test_mode eb2

# Diabetes
python main_Diabetes.py -e Diabetes_debug_130 --Diabetes_train_mode eb1 --Diabetes_test_mode eb2
python main_Diabetes.py -e Diabetes_debug_130 --Diabetes_train_mode eb1 --Diabetes_test_mode eb2

# Type I
python main_Diabetes.py --e 321_I_2 --bias_type I --minority young --minority_size 100 

# Type II 
python main_Diabetes.py -e 322_II --bias_type II --Diabetes_train_mode eb1 --Diabetes_test_mode eb2