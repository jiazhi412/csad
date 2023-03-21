from __future__ import print_function
import os
from itertools import product
from re import escape

if not os.path.exists('outputs'):
    os.makedirs('outputs')
# if not os.path.exists('errors'):
#     os.makedirs('errors')

# parameters
command_template = 'python main_German.py --exp_name {} --German_train_mode {} --German_test_mode {} --alpha {} --tau {} --lambda_ {}'
p1 = ['German_CSAD_paper_130']
p2 = ['eb1', 'eb2']
p3 = ['eb1', 'eb2', 'unbiased']
p4 = [1]
p5 = [10]
p6 = [1]

for p11, p22, p33, p44, p55, p66 in product(p1, p2, p3, p4, p5, p6):
    command = command_template.format(p11, p22, p33, p44, p55, p66)
    job_name = f'{p11}-{p22}-{p33}-{p44}-{p55}-{p66}'
    bash_file = '{}.sh'.format(job_name)
    with open( bash_file, 'w' ) as OUT:
        OUT.write('#!/bin/bash\n')
        OUT.write('#SBATCH --job-name={} \n'.format(job_name))
        OUT.write('#SBATCH --ntasks=1 \n')
        OUT.write('#SBATCH --account=other \n')
        OUT.write('#SBATCH --qos=flexible \n')
        # OUT.write('#SBATCH --qos=premium \n')
        OUT.write('#SBATCH --partition=ALL \n')
        OUT.write('#SBATCH --cpus-per-task=4 \n')
        OUT.write('#SBATCH --gres=gpu:1 \n')
        OUT.write('#SBATCH --mem=16G \n')
        OUT.write('#SBATCH --time=5-00:00:00 \n')
        # OUT.write('#SBATCH --exclude=vista18 \n')
        # OUT.write('#SBATCH --nodelist=vista[06,07] \n')
        # OUT.write('#SBATCH --exclude=vista[01,02,03,04,05,08,09,12,14] \n')
        OUT.write('#SBATCH --output=outputs/{}.out \n'.format(job_name))
        OUT.write('#SBATCH --error=outputs/{}.out \n'.format(job_name))
        # OUT.write('#SBATCH --error=errors/{}.out \n'.format(job_name))
        OUT.write('source ~/.bashrc\n')
        OUT.write('echo $HOSTNAME\n')
        OUT.write('echo "total gpu resources allocated: "$CUDA_VISIBLE_DEVICES\n')
        OUT.write('conda activate pytorch\n')
        OUT.write(command)

    qsub_command = 'sbatch {}'.format(bash_file)
    os.system( qsub_command )
    os.system('rm -f {}'.format(bash_file))
    print( qsub_command )
    print( 'Submitted' )


p2 = ['eb1_balanced', 'eb2_balanced']
p3 = ['eb1_balanced', 'eb2_balanced', 'balanced']

for p11, p22, p33, p44, p55, p66 in product(p1, p2, p3, p4, p5, p6):
    command = command_template.format(p11, p22, p33, p44, p55, p66)
    job_name = f'{p11}-{p22}-{p33}-{p44}-{p55}-{p66}'
    bash_file = '{}.sh'.format(job_name)
    with open( bash_file, 'w' ) as OUT:
        OUT.write('#!/bin/bash\n')
        OUT.write('#SBATCH --job-name={} \n'.format(job_name))
        OUT.write('#SBATCH --ntasks=1 \n')
        OUT.write('#SBATCH --account=other \n')
        OUT.write('#SBATCH --qos=flexible \n')
        # OUT.write('#SBATCH --qos=premium \n')
        OUT.write('#SBATCH --partition=ALL \n')
        OUT.write('#SBATCH --cpus-per-task=4 \n')
        OUT.write('#SBATCH --gres=gpu:1 \n')
        OUT.write('#SBATCH --mem=16G \n')
        OUT.write('#SBATCH --time=5-00:00:00 \n')
        # OUT.write('#SBATCH --nodelist=vista[06,07] \n')
        # OUT.write('#SBATCH --exclude=vista18 \n')
        # OUT.write('#SBATCH --exclude=vista[06,07,10,11,13,17-20] \n')
        # OUT.write('#SBATCH --exclude=vista[01,02,03,04,05,08,09,12,14] \n')
        OUT.write('#SBATCH --output=outputs/{}.out \n'.format(job_name))
        OUT.write('#SBATCH --error=errors/{}.out \n'.format(job_name))
        OUT.write('source ~/.bashrc\n')
        OUT.write('echo $HOSTNAME\n')
        OUT.write('echo "total gpu resources allocated: "$CUDA_VISIBLE_DEVICES\n')
        OUT.write('conda activate pytorch\n')
        OUT.write(command)

    qsub_command = 'sbatch {}'.format(bash_file)
    os.system( qsub_command )
    os.system('rm -f {}'.format(bash_file))
    print( qsub_command )
    print( 'Submitted' )
