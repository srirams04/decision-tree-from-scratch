"""
You can add your own functions here according to your decision tree implementation.
There is no restriction on following the below template, these fucntions are here to simply help you.
"""

from typing import Literal
import pandas as pd
import numpy as np

def check_ifreal(y: pd.Series) -> bool:
    """
    Function to check if the given series has real or discrete values
    """
    try:
        return any(y % 1 != 0)  # True if any value has a non-zero decimal part
    except TypeError:
        return False 



def entropy(Y: pd.Series) -> float:
    """
    Calculate entropy for a categorical variable.

    Parameters:
    - Y: pd.Series of categorical data

    Returns:
    - Entropy value
    """
 
    # Count the occurrences of each unique value in the series
    value_counts = Y.value_counts()

    # Calculate the probabilities of each unique value
    probabilities = value_counts / len(Y)

    nonzero_probabilities = probabilities[probabilities > 0]

    # Calculate entropy using the formula: H(S) = -p1*log2(p1) - p2*log2(p2) - ...
    entropy_value = -np.sum(nonzero_probabilities * np.log2(nonzero_probabilities))

    return entropy_value
    

def gini_index(Y: pd.Series) -> float:
    """
    Function to calculate the gini index
    """
    gini = 1
    for value in np.unique(Y) :
        p = (np.sum(Y == value))/(np.size(Y)) # prob of value in series
        if p>0:
          gini -= p**2
    return gini


def information_gain(Y: pd.Series, attr: pd.Series, criterion: Literal["information_gain", "gini_index"]) -> float:
    """
    Function to calculate the information gain
    Y = target attr, attr = attr i am splitting on
    """
    # Real output ==> Use MSE/Variance
    # Case 1: Real Input Real Output
    if check_ifreal(attr) and check_ifreal(Y):
        parent_variance = np.var(Y)
        left_variance = np.var(Y[attr <= attr.mean()])
        right_variance = np.var(Y[attr > attr.mean()])
        weights = [len(Y[attr <= attr.mean()])/len(Y), len(Y[attr > attr.mean()])/len(Y)] 
        weighted_variance = weights[0] * left_variance + weights[1] * right_variance
        return parent_variance - weighted_variance

    # Case 2: Discrete Input Real Output
    elif not check_ifreal(attr) and check_ifreal(Y):
        parent_variance = np.var(Y)
        uniq_attr = np.unique(attr)
        weighted_variances = 0
        for attribute in uniq_attr:
            Y_filtered = Y[attr == attribute]
            weight = len(Y_filtered)/len(Y)
            weighted_variances += weight * (np.var(Y_filtered))
        return(parent_variance - weighted_variances)
    

    # Discrete output ==> Use Entropy or Gini Index
    # Case 3: Real Input Discrete Output
    elif check_ifreal(attr) and not check_ifreal(Y):
        parent_impurity = entropy(Y) if criterion == "information_gain" else gini_index(Y)
        threshold = attr.mean()
        values = [attr <= threshold, attr > threshold]  # Discretize the feature
        weights = [len(Y[attr <= threshold]) / len(Y), len(Y[attr > threshold]) / len(Y)]
        weighted_impurities = 0
        for i in range(2):
            child_impurity = entropy(Y[values[i]]) if criterion == "information_gain" else gini_index(Y[values[i]])
            weighted_impurities += weights[i] * child_impurity
        
        return parent_impurity - weighted_impurities

    # Case 4: Discrete Input Discrete Output
    else:
        parent_impurity = entropy(Y) if criterion == "information_gain" else gini_index(Y)
        uniq_attr = np.unique(attr)
        weighted_impurities = 0
        for attribute in uniq_attr:
            Y_filtered = Y[attr == attribute]
            weight = len(Y_filtered)/len(Y)
            child_impurity = entropy(Y_filtered) if criterion == "information_gain" else gini_index(Y_filtered)
            weighted_impurities += weight * child_impurity

        return parent_impurity - weighted_impurities
    
        
def opt_split_attribute(X: pd.DataFrame, y: pd.Series, criterion: Literal["information_gain", "gini_index"], features: pd.Series):
    
    max_gain = -float('inf') 
    best_feature = None

    for feature in features:
        attr = X[feature]
        gain = information_gain(y, attr, criterion)  
        if gain > max_gain:
            max_gain = gain
            best_feature = feature

    return best_feature, max_gain


def split_data(X: pd.DataFrame, y: pd.Series, attribute: str, value: any) -> tuple[pd.DataFrame, pd.Series, pd.DataFrame, pd.Series]:
    
    if check_ifreal(X[attribute]): # Real Input
        mask = X[attribute] <= value
    else: # Discrete Input
        mask = X[attribute] == value    

    X_left = X[mask]
    y_left = y[mask]
    X_right = X[~mask]
    y_right = y[~mask]

    return X_left, y_left, X_right, y_right
