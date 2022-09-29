from __future__ import print_function
import os
from itertools import product
from re import escape

if not os.path.exists('outputs'):
    os.makedirs('outputs')
if not os.path.exists('errors'):
    os.makedirs('errors')

# parameters
command_template = 'python main_CelebA.py --exp_name {}  --attr {} --eval_mode {} --alpha {} --tau {} --lambda_ {}'
p1 = ['CelebA_paper']
p2 = ['Heavy_Makeup', 'Blond_Hair']
p3 = ['unbiased', 'conflict', 'unbiased_ex', 'conflict_ex', 'conflict_pp', 'conflict_pp_ex']
p4 = [1]
p5 = [10]
p6 = [1]

for p1, p2, p3, p4, p5, p6 in product(p1, p2, p3, p4, p5, p6):
    command = command_template.format(p1, p2, p3, p4, p5, p6)
    job_name = f'{p1}-{p2}-{p3}-{p4}-{p5}-{p6}'
    bash_file = '{}.sh'.format(job_name)
    with open( bash_file, 'w' ) as OUT:
        OUT.write('#!/bin/bash\n')
        OUT.write('#SBATCH --job-name={} \n'.format(job_name))
        OUT.write('#SBATCH --ntasks=1 \n')
        OUT.write('#SBATCH --account=other \n')
        OUT.write('#SBATCH --qos=premium \n')
        OUT.write('#SBATCH --partition=ALL \n')
        OUT.write('#SBATCH --cpus-per-task=4 \n')
        OUT.write('#SBATCH --gres=gpu:1 \n')
        OUT.write('#SBATCH --mem=16G \n')
        OUT.write('#SBATCH --time=5-00:00:00 \n')
        OUT.write('#SBATCH --exclude=vista[06,07,10,11,13,17-20] \n')
        # OUT.write('#SBATCH --exclude=vista18 \n')
        OUT.write('#SBATCH --output=outputs/{}.out \n'.format(job_name))
        OUT.write('#SBATCH --error=errors/{}.out \n'.format(job_name))
        OUT.write('source ~/.bashrc\n')
        OUT.write('conda activate vae\n')
        OUT.write(command)

    qsub_command = 'sbatch {}'.format(bash_file)
    os.system( qsub_command )
    os.system('rm -f {}'.format(bash_file))
    print( qsub_command )
    print( 'Submitted' )
