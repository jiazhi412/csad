# given pretrain
python main.py -e csad0020 --color_var 0.02 --checkpoint baseline/pretraincheckpoint_step_0000.pth --alpha 1 --tau 10 --lambda_ 1 

python main.py -e debug --color_var 0.02 

# own train
python main.py -e 0 --color_var 0.02 --alpha 1 --tau 10 --lambda_ 1



# CelebA
python main_CelebA.py