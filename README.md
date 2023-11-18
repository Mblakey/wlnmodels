# WLNModels
Models of WLN, requries the wln parser to be downloaded and built, which can be found [here](https://github.com/Mblakey/wiswesser). This repo uses machine learning
uses Wiswesser Line Notation (WLN) as a sequence descriptor. <br>

This project holds both Finite State Machine assisted Q-learning for steering molecular generation, alongside various chemical experiments using a previously 
unavaliable method for representing compounds. 

## LLFSM
Recurrent network that uses the FSM built from `wlngrep` to ensure any generated sequence is syntactically valid. FSM branches are used to recalculate character predictions
obtained from traning a GRU based system to predict the next WLN character. 

