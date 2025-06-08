import pandas as pd

def pandas_loader(data_path, datetime_column = None):
    if datetime_column is None:
        return pd.read_csv(data_path)
    else:
        return pd.read_csv(data_path, parse_dates= [datetime_column])