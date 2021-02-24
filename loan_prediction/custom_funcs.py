from sklearn.model_selection import StratifiedShuffleSplit


def initial_sss(df, label, test_size, out_file):

    sss = StratifiedShuffleSplit(n_splits=1, test_size=test_size,
                                 random_state=42)

    print(f'Spliting data: \n With shape of: {df.shape} \n Label being: {label} \n Output path: {out_file}')

    for train_index, test_index in sss.split(df, df[label]):
        train_df = df.iloc[train_index]
        test_df = df.iloc[test_index]

    print(f'Train shape: {train_df.shape}')
    print(f'Test shape: {test_df.shape}')

    train_df.to_csv(out_file / 'train_df.csv', index=False)
    test_df.drop(label, 1).to_csv(out_file / 'test_df.csv', index=False)
    test_df[label].to_csv(out_file / 'test_df_y_true.csv', index=False)

    print('Split successful')