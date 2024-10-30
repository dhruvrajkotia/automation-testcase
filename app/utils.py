
import numpy as np

def convert_numpy_types(data):
    if isinstance(data, np.bool_):  # Convert numpy bools to Python bools
        return bool(data)
    elif isinstance(data, np.ndarray):
        return data.tolist()  # Convert numpy arrays to lists
    elif isinstance(data, dict):
        return {key: convert_numpy_types(value) for key, value in data.items()}
    elif isinstance(data, list):
        return [convert_numpy_types(item) for item in data]
    else:
        return data

