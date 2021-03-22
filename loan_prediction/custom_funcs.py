import itertools 

from sklearn.model_selection import StratifiedShuffleSplit
from sklearn.preprocessing import OneHotEncoder
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
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
    print(f'Dtype: {df[col].dtype}')
    print(f'Null values: {df[col].isnull().sum()}')
    print(f'Unique: {df[col].nunique()}')

    col_val_counts = df[col].value_counts()[:show_vals]
    print(f'Vals: \n {col_val_counts}')
    top_vals = col_val_counts[:show_vals].index
    for val in top_vals:
        print(f'Current category: {val}')
        print(df[df[col] == val][label].value_counts(normalize=True))



def plot_confusion_matrix(cm,
                          target_names,
                          title='Confusion matrix',
                          cmap=None,
                          normalize=False):
    """
    given a sklearn confusion matrix (cm), make a nice plot

    Arguments
    ---------
    cm:           confusion matrix from sklearn.metrics.confusion_matrix

    target_names: given classification classes such as [0, 1, 2]
                  the class names, for example: ['high', 'medium', 'low']

    title:        the text to display at the top of the matrix

    cmap:         the gradient of the values displayed from matplotlib.pyplot.cm
                  see http://matplotlib.org/examples/color/colormaps_reference.html
                  plt.get_cmap('jet') or plt.cm.Blues

    normalize:    If False, plot the raw numbers
                  If True, plot the proportions

    Usage
    -----
    plot_confusion_matrix(cm           = cm,                  # confusion matrix created by
                                                              # sklearn.metrics.confusion_matrix
                          normalize    = True,                # show proportions
                          target_names = y_labels_vals,       # list of names of the classes
                          title        = best_estimator_name) # title of graph

    Citiation
    ---------
    http://scikit-learn.org/stable/auto_examples/model_selection/plot_confusion_matrix.html

    """


    accuracy = np.trace(cm) / float(np.sum(cm))
    misclass = 1 - accuracy

    if cmap is None:
        cmap = plt.get_cmap('Blues')

    plt.figure(figsize=(8, 6))
    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(title)
    plt.colorbar()

    if target_names is not None:
        tick_marks = np.arange(len(target_names))
        plt.xticks(tick_marks, target_names, rotation=45)
        plt.yticks(tick_marks, target_names)

    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]


    thresh = cm.max() / 1.5 if normalize else cm.max() / 2
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        if normalize:
            plt.text(j, i, "{:0.4f}".format(cm[i, j]),
                     horizontalalignment="center",
                     color="white" if cm[i, j] > thresh else "black")
        else:
            plt.text(j, i, "{:,}".format(cm[i, j]),
                     horizontalalignment="center",
                     color="white" if cm[i, j] > thresh else "black")


    plt.tight_layout()
    plt.ylabel('True label')
    plt.xlabel('Predicted label\naccuracy={:0.4f}; misclass={:0.4f}'.format(accuracy, misclass))
    plt.show()

    
"""
DATAFRAME PREPROCESSING:
"""


def drop_na(df):
    df = df.dropna(thresh=24)
    df = df.dropna(subset=['Name', 'City', 'State', 'Bank', 'BankState',
                           'NewExist', 'RevLineCr', 'LowDoc', 'DisbursementDate'])
    return df


def drop_col(df, col):
    df = df.drop(col, 1)
    return df


def clean_date_year(x):
    """
    This function is to be used in pandas.Series.apply().It takes in date
    as string and depending on the last two numbers(year) it adds 20 or 19
    before them and creates full year f.e 2011/1994, for pd.to_datetime to work
    :param x: date as string 
    :return: transformed x 
    """
    year = int(x[-2:])

    if year < 15:
        first_string = x[:-2]
        second_string = x[-2:]
        x = first_string + '20' + second_string
    
    elif year > 15: 
        first_string = x[:-2]
        second_string = x[-2:]
        x = first_string + '19' + second_string
        
    return x 


def date_to_datetime(df, col):
    """
    This function at fisrt cleans the data by adding century before year, 
    adding 0s at the start of
    value bcs pythons datetime string expects day to be zero padded
    decimal.Then it transforms the whole column to datetime dtype.
    Then it removes all rows with year under 1970.
    :param df: pandas.DataFrame
    :param col: col which to transform
    :return: df with tranformed col
    """

    df[col] = df[col].apply(clean_date_year)

    df[col] = df[col].apply(lambda x: 
                            '0' + x if x[1] == '-'
                            else x)

    df[col] = pd.to_datetime(df[col], format='%d-%b-%Y')
    df = df[df[col] > '1970']

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
    labels as Other, then it tranforms col to int64.
    :param df: pd.DAtaFrame
    :return:transformed df
    """

    df['NAICS'] = df['NAICS'].apply(lambda x: str(x)[:2])

    small_naics_ix = df['NAICS'].value_counts()[-6:].index
    df['NAICS'] = df['NAICS'].apply(lambda x:
                                    'Other' if x in small_naics_ix
                                    else x)

    # df['NAICS'] = df['NAICS'].astype('int64')                                    

    return df


"""
ApprovalFY COLUMN PREPROCESSING 
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
    This funtcion removes all zeros bcs term cannot be 0, then it makes all
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
    for group_range in range_list:
        df[col] = df[col].apply(lambda x:
                                group_range[0] + 1 if (x < group_range[1]) and (x > group_range[0])
                                else x)

    return df


"""
 RevLineCr COLUMN PREPROCESSING 
"""


def clean_rev_line_cr(df):
    """
    This function replaces all 0s as N bcs they had similar chrgoff_rates,
    then it removes all values other than 'Y', 'N', 'T',
    then it label encodes those values as ints
    :param df: pandas.DataFrame
    :return: modified pandas.DataFrame
    """
    df['RevLineCr'] = df['RevLineCr'].replace({'0': 'N'})

    values = ['Y', 'N', 'T']
    df = df[df['RevLineCr'].isin(values) == True]

    df['RevLineCr'] = df['RevLineCr'].replace(['N', 'Y', 'T'], [0, 1, 2])

    return df


"""
 LowDoc COLUMN PREPROCESSING 
"""


def clean_low_doc(df):
    """
    This function removes all values other than 'Y', 'N',
    then it label encodes those values as ints
    :param df: pandas.DataFrame
    :return: modified pandas.DataFrame
    """
    values = ['Y', 'N']
    df = df[df['LowDoc'].isin(values) == True]

    df['LowDoc'] = df['LowDoc'].replace(['N', 'Y'], [0, 1])

    return df


"""
 MIS_Status(label) COLUMN PREPROCESSING 
"""


def transform_label(df, label):
    """
    This funtion replaces label values as True False and renames col to Default
    :param df: pandas.DataFrame
    :param label: name of column thats label
    :return: modified pandas.DataFrame
    """
    df[label] = df[label].replace({'P I F': False,
                                   'CHGOFF': True})

    df = df.rename(columns={label: 'Default'})

    return df


"""
 SBA_Appv COLUMN PREPROCESSING 
"""


def clean_money_amount_col(df, col):
    """
    This funtion removes cents from the end of each amount and removes dollar
    sign, then it
    replaces every comma with nothing and changes dtype to int64
    :param df: pandas.DataFrame
    :param col: column name to clean
    :return: modified pandas.DataFrame
    """
    df[col] = df[col].apply(lambda x: x[1:-4])

    df[col] = df[col].apply(lambda x: x.replace(',', ''))
    df[col] = df[col].astype('int64')

    return df


def test_money_col(df, col):
    """
    This funtion is to test column if clean_money_amount_col() function
    will work, it tests if every value begins with dollar sign and if
    it ends with '.00 '.Then it takes example and transforms it and prints
    the output.
    :param df: pandas.DataFrame
    :param col: column name to test
    :return: modified pandas.DataFrame
    """
    dollar_mask = df[col].str.startswith('$')
    if df.shape == df[dollar_mask].shape:
        print('Everything starts with $')
    else:
        print('Found some that dont start with$')
        print(df.shape)
        print(df[dollar_mask].shape)

    ending_mask = df[col].str.endswith('.00 ')
    if df.shape == df[ending_mask].shape:
        print('Everything ends with .00 ')
    else:
        print('Found some that dont end with .00 ')
        print(df.shape)
        print(df[ending_mask].shape)

    test_str = df.iloc[2][col]
    print(f'Test string: {test_str}')
    test_str = test_str[1:-4]
    test_str = int(test_str.replace(',', ''))
    print(f'Preprocessed test string: {test_str}')


"""
 Recession COLUMN PREPROCESSING 
"""


def create_recession_col(df):
    """
    This function creates column that is True if year is equal to years of
    recession
    :param df: pandas.DataFrame
    :return: modified pandas.DataFrame
    """
    recession_years = [2007, 2008, 2009]
    df['Recession'] = df['DisbursementDate'].apply(lambda x:
                                                   True if x.year in recession_years
                                                   else False)

    return df


"""
 days_to_disbursement COLUMN PREPROCESSING 
"""


def create_days_to_disbursement(df):
    """
    This function creates column that explains how much days it took for
    money to be disbursed.
    :param df: pandas.DataFrame
    :return: modified pandas.DataFrame
    """
    df['days_to_disbursement'] = (df['DisbursementDate'] - df['ApprovalDate']).dt.days
    df['days_to_disbursement'] = df['days_to_disbursement'][df['days_to_disbursement'] >= 0]

    df['days_to_disbursement'] = df['days_to_disbursement'].apply(lambda x:
                                                                  None if x > 2000
                                                                  else x)

    df = df.dropna(subset=['days_to_disbursement'])
    df['days_to_disbursement'] = df['days_to_disbursement'].astype('int64')

    return df


"""
Dates COLUMNs PREPROCESSING 
"""

def create_date_data(df):
    """
    This function creates year column from DisbursementDate
    calumn.
    :param df: pandas.DataFrame
    :return: modified pandas.DataFrame
    """
    df['disbursement_year'] = df['DisbursementDate'].dt.year


    return df


"""
DROPING COLUMNS
"""
def drop_cols_at_end(df):
    """
    This function drops all columns at the end of preprocessing.
    :param df: pandas.DataFrame
    :return: modified pandas.DataFrame
    """
    to_drop = ['LoanNr_ChkDgt', 'City', 'Zip', 'Bank', 'NewExist', 'ChgOffDate', 'DisbursementGross',
               'BalanceGross', 'ChgOffPrinGr', 'GrAppv', 'ApprovalDate',
               'ApprovalFY', 'DisbursementDate']
    df = drop_col(df, to_drop)

    return df


"""
OUTLIER REMOVAL
"""




def outlier_treatment(datacolumn, multiplier):
    """
    This funtion drops all columns at the end of preprocessing.
    :param datacolumn: column of dataframe as Series
    :param multiplier: number by which we multiply iqr, bigger the number less 
    outliers we remove(ussually we use 1.5)
    :return: lower and upper ranges of values we are keeping 
    """
    datacolumn = sorted(datacolumn)
    q1, q3 = np.percentile(datacolumn, [25,75])
    iqr = q3 - q1
    lower_range = q1 - (multiplier * iqr)
    upper_range = q3 + (multiplier * iqr)

    return lower_range,upper_range

def remove_outliers(df, multiplier):
    """
    This function removes outliers from selected columns based on iqr range 
    defined by outlier_treatment() funtion
    :param df: pandas.DataFrame
    :param multiplier: number by which we multiply iqr, bigger the number less 
    outliers we remove(ussually we use 1.5)
    :return: modified pandas.DataFrame without outliers 
    """
    outlier_cols = ['days_to_disbursement', 'SBA_Appv']

    for col in outlier_cols:
        lower_range, upper_range = outlier_treatment(df[col], multiplier)
        outliers = df.loc[(df[col] > upper_range) | (df[col] < lower_range)]
        outliers_indexes = outliers.index
        df = df.drop(outliers_indexes)

    return df 


"""
FINAL PIPELINE
"""


def data_transformer(df):
    """
    This function aplies all tranformation functions on dataframe.
    :param df: pandas.DataFrame
    :return: modified pandas.DataFrame
    """
    df = drop_na(df)
    df = get_endings(df)
    df = tranform_bank_state(df)
    df = transform_naics_col(df)
    df = date_to_datetime(df, 'ApprovalDate')
    df = clean_approval_fy_col(df)
    df = term_transformer(df)
    df = transform_noemp(df)

    create_jobs_range_list = [(30, df['CreateJob'].max() + 1),
                          (20, 30),
                          (15, 20)]
    df = group_values(df, 'CreateJob', create_jobs_range_list)

    ret_job_range_list = [(49, df['RetainedJob'].max() + 1),
                      (25, 50)]
    df = group_values(df, 'RetainedJob', ret_job_range_list)

    franchise_code_range_list = [(1, df['FranchiseCode'].max() + 1)]
    df = group_values(df, 'FranchiseCode', franchise_code_range_list)

    df = clean_rev_line_cr(df)
    df = clean_low_doc(df)
    df = date_to_datetime(df, 'DisbursementDate')
    df = clean_money_amount_col(df, 'SBA_Appv')
    df = create_recession_col(df)
    df = create_days_to_disbursement(df)
    df = create_date_data(df)
    df = df.reset_index(drop=True)
    try:
        df = transform_label(df, 'MIS_Status')
    except: print('Label not found')
    df = drop_cols_at_end(df)

    return df


def onehot_enc(df):
    """
    This function onehotencodes selected columns and adds names to new cols.
    :param df: pandas.DataFrame
    :return: modified pandas.DataFrame
    """
    cat_cols = ['name_end', 'State', 'NAICS', 'FranchiseCode', 'UrbanRural', 'RevLineCr']

    ohe = OneHotEncoder()
    enc_arr = ohe.fit_transform(df[cat_cols]).toarray()
    col_names = ohe.get_feature_names(cat_cols)
    enc_df = pd.DataFrame(enc_arr, columns=col_names)

    df = df.reset_index(drop=True)
    df = pd.concat([df, enc_df], axis=1)
    df = df.drop(cat_cols, axis=1)

    return df 


def transform_train(X, y, oversampler):
    """
    This function oversamples training data using given oversampler from 
    imblearn module.
    :param X: independent features as DataFrame or np.array
    :param X: dependent feature as np.array or pd.Series
    :return: modified X, y
    """
    print(f'Before: {y.value_counts()}')
    X_res, y_res = oversampler.fit_resample(X, y)
    X = pd.DataFrame(X_res)
    y = pd.Series(y_res)
    print(f'After: {y.value_counts()}')

    return X, y































    