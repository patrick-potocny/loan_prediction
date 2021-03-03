from sklearn.model_selection import StratifiedShuffleSplit
import pandas as pd
import numpy as np
"""
UNIVERSAL FUNCS
"""


def initial_sss(df, label, test_size, out_file):
    """
    This funtion splits data into three parts:
     - train_df(dataframe containing both train data and labels)
     - test_df(dataframe containing just test data to be passed to predict on)
     - test_df_y_true(labels for test_df)
    Function uses StratifiedShuffleSplit from sklearn.model_selection
    :param df: pandas.DataFrame to be split
    :param label: name of the label column
    :param test_size: int from 0 to 1, percentage of data thats gonna be in test_df
    :param out_file: pathlib.Path() object, absolute path to directory where the split data shoould go
    :return: None
    """

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


def basic_cat_col_data(df, col, label='MIS_Status', show_vals=10):
    """
    This function basic info about categorical column
    :param df: pandas DataFrame
    :param col: column name to display info about
    :param label: label of df, Default MIS_Status
    :param show_vals: how much values to show in value counts, Default 10
    :return:
    """
    print(f'Null values: {df[col].isnull().sum()}')
    print(f'Unique: {df[col].nunique()}')

    col_val_counts = df[col].value_counts()[:show_vals]
    print(f'Vals: \n {col_val_counts}')
    top_vals = col_val_counts[:show_vals].index
    for val in top_vals:
        print(f'Current category: {val}')
        print(df[df[col] == val][label].value_counts(normalize=True))


"""
DATAFRAME PREPROCESSING:
"""


def drop_na(df):
    df = df.dropna(thresh=24)
    df = df.dropna(subset=['Name', 'City', 'State', 'Bank', 'BankState',
                           'NewExist', 'RevLineCr', 'LowDoc', 'DisbursementDate'])
    return df


"""
Name COLUMN PREPROCESSING:
"""


def get_endings(df):
    """
    This function changes Name col to just ending of the name if its in most
     used endings otherwise its Other
    :param df: pandas DataFrame to change
    :return: modified dataframe
    """
    most_used_ends = ['INC', 'INC.', 'LLC', 'Inc.', 'TION', 'Inc', 'PANY']

    for end in most_used_ends:
        df['Name'] = df['Name'].apply(lambda x:
                                      end if str(x).endswith(end)
                                      else x)

    df['Name'] = df['Name'].apply(lambda x: x if str(x) in most_used_ends
                                  else 'Other')

    df = df.rename(columns={'Name': 'name_end'})

    return df


"""
City COLUMN PREPROCESSING:
"""


def drop_col(df, col):
    df = df.drop(col, 1)
    return df


"""
State COLUMN PREPROCESSING:
"""


def get_states_rates(df, label):
    """
    This function returns dataframe with chrgoff_rate for each state
    calculated from entire df and its label
    :param df:
    :param label:
    :return: states_rates
    """
    states = df['State'].value_counts().index
    chrgoff_rates = []

    for state in states:
        val_counts = df[df['State'] == state][label].value_counts(normalize=True)

        chrgoff_rate = val_counts[1]
        chrgoff_rates.append(chrgoff_rate)

    state_rates = pd.DataFrame({'state': states,
                                'chrgoff_rate': chrgoff_rates})

    state_rates = state_rates.sort_values(by='chrgoff_rate', ascending=True)

    state_rates['chrgoff_rate'] = state_rates['chrgoff_rate'].apply(
        lambda x: round(x, 2))

    return state_rates


def group_rates(x):
    """
    This is custom funtion to be used in pandas.Series.apply() and returns
    series with values grouped based on similar rates
    :param x: element of series
    :return: group encoding for single element of series
    """

    if x < 0.11:
        group = 'u_11'
    elif 0.10 < x < 0.15:
        group = 'u_15'
    elif 0.14 < x < 0.19:
        group = 'u_19'
    elif 0.18 < x < 0.23:
        group = 'u_23'
    else:
        group = 'u_max'
    return group


def get_state_rate_groups(states_rates):
    """
    This function returns lists of states in groups based on chrgoff_rate,
    f.e. u_11_states means all states with chrgoff_rate smaller than 11%
        u_max_states means all states with worst chrgoff_rates starting from
        last groups upper threshold (23 in this case)
    :param states_rates: pandas.Dataframe returned from get_states_rates()
    :return: lists of 5 groups based on similar chrgoff_rates
    """
    states_rates['chrgoff_rate'] = states_rates['chrgoff_rate'].apply(
        group_rates)

    u_11_states = states_rates[states_rates['chrgoff_rate'] == 'u_11']
    u_11_states = u_11_states['state'].tolist()

    u_15_states = states_rates[states_rates['chrgoff_rate'] == 'u_15']
    u_15_states = u_15_states['state'].tolist()

    u_19_states = states_rates[states_rates['chrgoff_rate'] == 'u_19']
    u_19_states = u_19_states['state'].tolist()

    u_23_states = states_rates[states_rates['chrgoff_rate'] == 'u_23']
    u_23_states = u_23_states['state'].tolist()

    u_max_states = states_rates[states_rates['chrgoff_rate'] == 'u_max']
    u_max_states = u_max_states['state'].tolist()

    return u_11_states, u_15_states, u_19_states, u_23_states, u_max_states


def transform_state_col(x, u_11_states, u_15_states, u_19_states,
                        u_23_states, u_max_states):
    """
    This is custom function to be used in pandas.Series.apply(), it
    tranforms every state abbrev to category f.e(u_11, u_15)
    based on its states chrgoff_rate

    :param x: single element of the series
    :param u_11_states: list of states from states_to_rate_categories()
    :param u_15_states: list of states from states_to_rate_categories()
    :param u_19_states: list of states from states_to_rate_categories()
    :param u_23_states: list of states from states_to_rate_categories()
    :param u_max_states: list of states from states_to_rate_categories()
    :return: returns group assigned to element
    """
    group = ''

    if x in u_11_states:
        group = 'u_11'

    elif x in u_15_states:
        group = 'u_15'

    elif x in u_19_states:
        group = 'u_19'

    elif x in u_23_states:
        group = 'u_23'

    elif x in u_max_states:
        group = 'u_max'

    return group


def states_to_rate_categories(df, label):
    """
    This function puts together other preprocessing functions and
    transforms state names to groups based on their chrgoff_rates(label),
    at the and deletes nan values
    :param df: pandas.DataFrame with data to tranform
    :param label: label of data
    :return: returns dataframe with tranformed State column
    """

    states_rates = get_states_rates(df, label)

    df['State'] = df['State'].apply(transform_state_col,
                                    args=get_state_rate_groups(states_rates))

    df = df[df['State'] != '']

    return df


"""
BankState COLUMN PREPROCESSING:
"""


# not used for now
# def replace_small_states(df):
#     other_replace_dict = {'PR':'Other',
#                           'GU':'Other',
#                           'AN':'Other',
#                           'EN':'Other',
#                           'VI':'Other'}
#     df['BankState'] = df['BankState'].replace(other_replace_dict)
#
#     return df


def tranform_bank_state(df):
    """
    This function transforms BankState column to same_state which can be:
    - True = Bussiness state and bank state are equal
    - False = when they are not the same
    :param df: dataframe containing BankStete column
    :return: df with Bankstate tranformed to same_state
    """
    df['BankState'] = df['State'] == df['BankState']
    df = df.rename(columns={'BankState': 'same_state'})

    return df


"""
NAICS COLUMN PREPROCESSING 
"""

def transform_naics_col(df):
    """
    This function tranforms NAICS column, it takes just the two first
    two numbers of the whole code and last 6 less represented values
    labels as Other
    :param df: pd.DAtaFrame
    :return:transformed df
    """

    df['NAICS'] = df['NAICS'].apply(lambda x: str(x)[:2])

    small_naics_ix = df['NAICS'].value_counts()[-6:].index
    df['NAICS'] = df['NAICS'].apply(lambda x:
                                    'Other' if x in small_naics_ix
                                    else x)

    return df


"""
ApprovalDate COLUMN PREPROCESSING 
"""


def approval_date_to_datetime(df):
    """
    This function at fisrt cleans the data by adding 0s at the start of
    value bcs pythons datetime string expects day to be zero padded
    decimal.Then it transforms the whole column to datetime dtype.
    :param df: pandas.DataFrame
    :return: df with tranformed col
    """
    df['ApprovalDate'] = df['ApprovalDate'].apply(lambda x:
                                                  '0'+x if x[1] == '-'
                                                  else x)

    df['ApprovalDate'] = pd.to_datetime(df['ApprovalDate'])
    return df


"""
ApprovalDate COLUMN PREPROCESSING 
"""


def clean_str(x):
    """
    This function is to be used ind pd.Series.apply(), and it cleans col
    from As.
    :param x: element passed by apply
    :return: element
    """
    if isinstance(x, str):
        x = x.replace('A', '')
    return x


def clean_approval_fy_col(df):
    """
    This function cleans approval_fy and makes them int64 dtype.
    :param df: pandas.DataFrame
    :return: df with tranformed col
    """
    df['ApprovalFY'] = df['ApprovalFY'].apply(clean_str).astype('int64')

    return df


"""
Term COLUMN PREPROCESSING 
"""


def term_transformer(df):
    """
    This funtion removes all zeros bcs term cannot be 0, then it makes all
    terms bigger than 300 into single value
    :param df: pandas.DataFrame
    :return: df with tranformed col
    """
    df = df[df['Term'] != 0]

    df['Term'] = df['Term'].apply(lambda x:
                                  310 if x > 300
                                  else x)

    return df


"""
 NoEmp COLUMN PREPROCESSING 
"""


def transform_noemp(df):
    """
    This funtion groups the higher numbers of employees into same values
    bcs of their low frequency, this function should be replaced by
    group_values()
    :param df: pandas.DataFrame
    :return: df with tranformed col
    """
    df['NoEmp'] = df['NoEmp'].apply(lambda x:
                                    110 if x > 100
                                    else x)

    df['NoEmp'] = df['NoEmp'].apply(lambda x:
                                    90 if (x < 100) and x > 90
                                    else x)

    df['NoEmp'] = df['NoEmp'].apply(lambda x:
                                    80 if (x < 90) and x > 80
                                    else x)
    return df


"""
 CreateJob COLUMN PREPROCESSING 
"""


def group_values(df, col, range_list):
    """
    This function groups values based on predefined ranges, can be used
    for values with low frequencies to get rif of outliers.
    :param df: pandas.DataFrame
    :param col: string, column name to be tranformed
    :param range_list: list of tuples, each tuple represents range,
    (lower_range, upper_range) - both are exclusive
    f.e [(0, 11)-from 1 to 10, (10, 21) - from 11 to 20]
    :return:
    """
    for range in range_list:
        df[col] = df[col].apply(lambda x:
                                range[0] if (x < range[1]) and (x > range[0])
                                else x)

    return df


"""
  COLUMN PREPROCESSING 
"""