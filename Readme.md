# NeWise
A prototype verification tool.
## Installation
***
We first provide scripts that will install all the necessary dependencies.
```
. install.sh
```

When all the necessary dependencies are installed, there will be "The enviroment has been deployed!".

## How to Run
***

```
. run.sh
```

Table results will be saved in 'results/'. The figures will be saved in 'figs/'.

All the scripts were tested on a workstation running Ubuntu 18.04.

Note that when using Algorithm 1 to calculate the certified lower bound, the images were taken randomly, so the re-run results will be slightly different from those in the paper. However, the results are still consistent with the conclusion in the paper that the approximation computed by Algorithm 1 is the optimal approximation for a neural network containing only one hidden layer.