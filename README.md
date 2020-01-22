# DAC
Code/Supplementary for the ECAI2020 paper<BR>
**Dynamic Algorithm Configuration:<BR>Laying the foundation of a new Framework**

If you used the presented benchmarks or framework in one of your research projects, please cite us:

    @inproceedings{biedenkapp-ecai20,
      author    = {A. Biedenkapp and H. F. Bozkurt and T. Eimer and F. Hutter and M. Lindauer},
      title     = {Dynamic Algorithm Configuration: Foundation of a New Meta-Algorithmic Framework},
      booktitle = {Proceedings of the Twenty-fourth European Conference on Artificial Intelligence ({ECAI}'20)},
      year = {2020},
      month     = jun,
    }


## Installation using Anaconda
1. conda create -n dac python=3.7
2. conda activate dac
3. cat requirements.txt | xargs -L 1 -n 1 pip install
4. python setup.py install
5. (*optional*) conda install jupyter 

## Example calls
* Train 0.1-greedy tabular Q-learning on 1D-sigmoids with trinary actions for 10<sup>4</sup> episodes:<BR>
```python dac/train/train_other.py --seed 0 -r 1 -n 10000 --epsilon_decay const -e 0.1 -l 1. --env 1D3M --out-dir 1D-trinary-action-Sigmoid```
* Train DDQN on 1D-sigmoids with trinary actions for 10<sup>5</sup>/10<sup>4</sup> steps/episodes:<BR>
```python dac/train/train_chainer_agent_on_toy.py --eval-n-runs 10 --eval-interval 10 --checkpoint_frequency 1000 --outdir 1D-trinary-action-Sigmoid/DQN --seed 0 --scenario 1D3M --steps 100000```
* Run PS-SMAC on 1D-sigmoids with trinary actions for 10<sup>4</sup> episodes:
  1. Find well performing parameter sequence:<BR>
  ```python dac/train/train_other.py --seed 0 -r 1 -n 10000 --epsilon_decay const -e 0.1 -l 1. --env 1D3M --out-dir 1D-trinary-action-Sigmoid --bo```
  2. Validate the found sequence:<BR>
  ```python dac/train/train_other.py --seed 0 -r 1 -n 10000 --epsilon_decay const -e 0.1 -l 1. --env 1D3M --out-dir 1D-trinary-action-Sigmoid --bo --validate-bo 1D-trinary-action-Sigmoid/smac3*/run*```

We provide a jupyter notebook with which you can inspect your own runs or the example results provided in example-results-1D-trinary-action-Sigmoid

## License
This repository is licensed under the [**Apache License 2.0**](LICENSE)

## Repository Structure
* cmds:<BR>
  Contains many more commands to run sigmoid experiments from the paper
* dac
  * envs:<BR>
    Contains all instance features and code for the presented white-box benchmarks
  * train:<BR>
    Contains code to train all the agents presented in the paper
* example-results-1D-trinary-action-Sigmoid
  * DQN: <BR>
    Result of the above example DQN call
  * smac3-output:<BR>
    Results of the call of PS-SMAC
  * tabular:<BR>
    Result of the above example tabular call
* Appendix.pdf<BR>
  The online appendix of the paper.
* example_plots.ipynb<BR>
  Jupyter-Notebook to plot the example results