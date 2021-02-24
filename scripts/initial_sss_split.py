from pathlib import Path

import pandas as pd

from loan_prediction.config import raw_data_path, split_dir, \
    split_test_y_true, split_test, split_train
from loan_prediction.custom_funcs import initial_sss


if split_test_y_true.exists() and split_test.exists() and split_train.exists():
    print(f'Split data is already present in: \n {split_dir}')
else:
    print('Proceeding ot split data: ')

    df = pd.read_csv(raw_data_path)
    label = 'MIS_Status'

    df = df[df['MIS_Status'].notna()]

    initial_sss(df, label, 0.2, split_dir)