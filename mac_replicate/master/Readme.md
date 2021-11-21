MAC replication - This is supposed be temporary folder. Trying a temporary
clean mac folder

### Download
https://github.com/AI4Finance-Foundation/Deep-Reinforcement-Learning-for-Automated-Stock-Trading-Ensemble-Strategy-ICAIF-2020

/Users/jvsingh/work/stanford/XCS229ii-ML-Strategy-RL/Project/xcs229ii-project/mac_replicate

### install brew if it is not installed
`ruby -e "$(curl -fsSL https://raw.githubusercontent.com/Homebrew/install/master/install)"`

### This uses openmpi
`brew install cmake openmpi`

### Create conda environment
Please change last line of `rlproject_env.yaml` as per your directory settings.

`conda env create --file rlproject_env.yaml`

`conda activate rlproject`

### In case you need to delete the created env

`conda deactivate`

`conda remove --name rlproject --all`
