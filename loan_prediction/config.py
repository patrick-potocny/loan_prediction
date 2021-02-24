from pathlib import Path

current_wd = Path(__file__).resolve()
current_wd = current_wd.parent.parent


raw_data_path = current_wd / 'data/raw/SBAnational.csv'


split_dir = current_wd / 'data' / 'split'
split_train = split_dir / 'train_df.csv'
split_test = split_dir / 'test_df.csv'
split_test_y_true = split_dir / 'test_df_y_true.csv'

predictions_csv = current_wd / 'data/predictions/predictions.csv'