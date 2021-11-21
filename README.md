# xcs229ii-project
Projects for xcs229ii-project [James/Benjamin/Jai Vrat Singh]


## Reproduce Results from DRL Ensemble Stock Trader
- Code: https://github.com/AI4Finance-Foundation/Deep-Reinforcement-Learning-for-Automated-Stock-Trading-Ensemble-Strategy-ICAIF-2020
- Paper: https://papers.ssrn.com/sol3/papers.cfm?abstract_id=3690996
- Blog: https://towardsdatascience.com/deep-reinforcement-learning-for-automated-stock-trading-f1dad0126a02

### issues while installing on Win
* DLL error solution: https://stackoverflow.com/questions/56433666/openai-spinning-up-problem-importerror-dll-load-failed-the-specified-procedur/58653569#58653569
* changed config/config.py line 23 
TRAINED_MODEL_DIR = f"trained_models/{now}" 
=> 
timestampStr = now.strftime("%d-%b-%Y %H%M%S")
TRAINED_MODEL_DIR = r"trained_models/"+timestampStr
* 
