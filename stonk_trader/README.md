Projects for xcs229ii-project [James/Benjamin/Jai Vrat Singh]

This is new space where we first work on the replicated copy of research done
in https://github.com/AI4Finance-Foundation/Deep-Reinforcement-Learning-for-Automated-Stock-Trading-Ensemble-Strategy-ICAIF-2020

Paper: https://papers.ssrn.com/sol3/papers.cfm?abstract_id=3690996



## Creating environment to run this
### 1. Install anaconda

### 2. Create environment using the yaml file
First check location of your anaconda3 folder. I have used $HOME/anaconda3/envs/rlproject where the environment will be created.
This is mentioned in last line of the rlproject_env.yaml file. Change it accordingly.

`conda env create --file rlproject_env.yaml`


### 3. Activate your environment
`conda activate rlproject`

(If you need to deactivate after activation, use this)
`conda deactivate`

### 4. Install Kernel
`python -m ipykernel install --user --name=rlproject`
