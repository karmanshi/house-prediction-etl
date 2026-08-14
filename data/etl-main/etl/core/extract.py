import pandas as pd
from .common import NoRecordFound


def data_extract(file_path):
    data_set = pd.read_csv(file_path)
    df = pd.DataFrame(data_set)

    row_count = df.shape[0]

    if row_count>0:
        return df
    else:
        raise NoRecordFound()