import yaml
from visibility.exception import VisibilityClimateException
from visibility.logger import logging
import os,sys
import numpy as np
import dill
import numpy as np
import pandas as pd


def read_yaml_file(file_path: str) -> dict:
    try:
        with open(file_path, "rb") as yaml_file:
            return yaml.safe_load(yaml_file)
    except Exception as e:
        raise VisibilityClimateException(e, sys) from e


def write_yaml_file(file_path: str, content: object, replace: bool = False) -> None:
    try:
        if replace:
            if os.path.exists(file_path):
                os.remove(file_path)
        os.makedirs(os.path.dirname(file_path), exist_ok=True)
        with open(file_path, "w") as file:
            yaml.dump(content, file)
    except Exception as e:
        raise VisibilityClimateException(e, sys)



def save_numpy_array_data(file_path: str, array: np.array):
    """
    Save numpy array data to file
    file_path: str location of file to save
    array: np.array data to save
    """
    try:
        dir_path = os.path.dirname(file_path)
        os.makedirs(dir_path, exist_ok=True)
        with open(file_path, "wb") as file_obj:
            np.save(file_obj, array)
    except Exception as e:
        raise VisibilityClimateException(e, sys) from e


def load_numpy_array_data(file_path: str) -> np.array:
    """
    load numpy array data from file
    file_path: str location of file to load
    return: np.array data loaded
    """
    try:
        with open(file_path, "rb") as file_obj:
            return np.load(file_obj)
    except Exception as e:
        raise VisibilityClimateException(e, sys) from e


def save_object(file_path: str, obj: object) -> None:
    try:
        logging.info("Entered the save_object method of MainUtils class")
        os.makedirs(os.path.dirname(file_path), exist_ok=True)
        with open(file_path, "wb") as file_obj:
            dill.dump(obj, file_obj)
        logging.info("Exited the save_object method of MainUtils class")
    except Exception as e:
        raise VisibilityClimateException(e, sys) from e


def load_object(file_path: str, ) -> object:
    try:
        if not os.path.exists(file_path):
            raise Exception(f"The file: {file_path} is not exists")
        with open(file_path, "rb") as file_obj:
            return dill.load(file_obj)
    except Exception as e:
        raise VisibilityClimateException(e, sys) from e
def write_schema_file(file_path_source,file_path_destination):
    try:
        data = pd.read_csv(file_path_source)
        column_types = data.dtypes.tolist()
        change_type = [str(i).replace('float64','float').replace('int64','int').replace('object','text') for i in column_types ]
        columns_with_datatype = []
        for idx,col in enumerate(data.columns):
            di ={}
            di[col] = change_type[idx]
            columns_with_datatype.append(di)
        content_dict = {}
        numerical_columns = []
        content_dict["columns"] = columns_with_datatype
        for d in columns_with_datatype:
            for key,val in d.items():
                if val=='int' or val =="float":
                    numerical_columns.append(key)
        content_dict['numerical_columns']= numerical_columns

        with open(file_path_destination, 'w') as file:
            yaml.dump(content_dict, file)
    except Exception as e:
        raise VisibilityClimateException(e,sys)
