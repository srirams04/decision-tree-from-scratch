## Decision Tree Classifier + Regressor Implementation

The decision tree implementation does not use existing machine learning libraries like scikit-learn. It is implemented from scratch using Python (NumPy and Pandas) and works for four cases: 
- Discrete features and discrete output
- Discrete features and real output
- Real features and discrete output
- Real features and real output
 
The decision tree can switch between GiniIndex or Information Gain (Entropy) as the criteria for splitting for *discrete* output, and uses Information Gain (MSE) as the criteria for splitting for *real* output.

- `metrics.py`: Contains the performance metrics functions.

- `usage.py`: This file can be run to check the implementation.

- tree (Directory): Module for decision tree.
  - `base.py` : Contains Decision Tree Class.
  - `utils.py`: Contains all utility functions.
  - `__init__.py`: **Do not edit this**


## Acknowledgement

This task was completed as part of the Machine Learning course at IIT Gandhinagar, under [Prof. Nipun Batra](https://nipunbatra.github.io/).
