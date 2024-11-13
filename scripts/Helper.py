from pandas.core.frame import DataFrame
import numpy as np
def ratio(X :DataFrame|np.ndarray) -> np.ndarray:
    x_array = np.array(X)
    return x_array[:,[0]] / x_array[:,[1]]

def ratio_column_name(function_transformer, feature_names_in) -> list:
    return [f'{feature_names_in[0]}_to_{feature_names_in[1]}_ratio'];

def percent_to_ratio(X: DataFrame|np.ndarray) -> np.ndarray:
    x_array = np.array(X)
    return x_array[:,[0]]/100

def percent_to_ratio_column_name(function_transfomer, feature_names_in) -> list:
    return [f'{feature_name_in}_ratio' for feature_name_in in feature_names_in]

def array_transformer(X: DataFrame|np.ndarray) -> np.ndarray:
    return np.array(X)

def array_transformer_column_names(function_transformer, feature_names_in) -> list:
    return feature_names_in