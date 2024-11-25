# import warnings # supress warnings
# warnings.filterwarnings('ignore')

import pickle

import numpy as np
import pandas as pd
from sklearn.preprocessing import OrdinalEncoder

from settings import DATA_DIR, MODEL_DIR, REMOVE_OUTLIERS, TARGET

# Display all columns
# pd.set_option('display.max_columns', None)
# pd.set_option('display.max_rows', None)


from settings import DEBUG  # isort:skip

DEBUG = True  # True # False # override global settings


def load_dataset(verbose=DEBUG):
    file_name = 'gym_churn_us.csv'
    data = pd.read_csv(DATA_DIR + file_name)

    if verbose:
        print(f' Loaded {data.shape[0]} records.')
        data.info()
        sample = 3
        print(f'\nDataset head({sample}):\n', data.head(sample).T.to_string())

    return data


def determine_outlier_thresholds_iqr(df, col_name, th1=0.25, th3=0.75):
    # for removing outliers using Interquartile Range or IQR
    quartile1 = df[col_name].quantile(th1)
    quartile3 = df[col_name].quantile(th3)
    iqr = quartile3 - quartile1
    upper_limit = quartile3 + 1.5 * iqr
    lower_limit = quartile1 - 1.5 * iqr
    return lower_limit, upper_limit


def determine_outlier_thresholds_sdm(df, col_name, scale):
    # for removing outliers using the Standard deviation method
    df_mean = df[col_name].mean()
    df_std = df[col_name].std()
    upper_limit = df_mean + scale * df_std
    lower_limit = df_mean - scale * df_std
    return lower_limit, upper_limit


def print_missing_values_table(data, na_name=False):
    na_columns = [col for col in data.columns if data[col].isnull().sum() > 0]
    n_miss = data[na_columns].isnull().sum().sort_values(ascending=False)
    ratio = (data[na_columns].isnull().sum() / data.shape[0] * 100).sort_values(ascending=False)
    missing_df = pd.concat([n_miss, np.round(ratio, 2)], axis=1, keys=['n_miss', 'ratio'])
    print(missing_df, end="\n")
    if na_name:
        return na_columns


def preprocess_df(data, verbose=DEBUG):
    print('\nPreprocessing data...')
    target = TARGET
    total_rows_number = data.shape[0]

    # normalizing column names - lowercase, no spaces
    data.columns = data.columns.str.lower().str.replace(" ", "_")
    columns = data.columns.to_list()

    # 1. drop useless columns
    useless_columns = []
    for col in useless_columns:
        if col in columns:  # exists in df
            if verbose:
                print(f'Dropping column {col}. Total rows: {total_rows_number}, unique {col.upper()}s: {data[col].nunique()}')
            data.drop([col], axis=1, inplace=True)

    # 2. inspect categorical columns
    categorical = data.dtypes[data.dtypes == 'object'].keys()
    if target in categorical:
        # specifics of some datasets, TARGET must be encoded to 0/1 BEFORE using OrdinalEncoder
        data.loc[data[target] == 'No', target] = 0
        data.loc[data[target] == 'Yes', target] = 1
        data[target] = data[target].astype(int)
        # update categorical
        categorical = data.dtypes[data.dtypes == 'object'].keys()
        if verbose:
            print(target, 'encoded')

    if verbose:
        if len(categorical):
            print(f'\nCategorical columns: {list(categorical)}')
            # print distribution for each
            for col in categorical:
                print('\n by', data[col].value_counts().to_string())

            if target in columns and len(data) > 10:
                corr = data.corr(numeric_only=True)[target]
                print(f'\nCorrelation to {target}:\n{corr.sort_values().to_string()}')
        else:
            print(f'\nNo categorical columns.')


    # 3. inspect missing values
    nan_cols = data.columns[data.isnull().any()].to_list()
    if verbose and nan_cols:
        # list of columns with missing values and its percentage
        print(f'\nColumns with nulls:\n{nan_cols}')
        print_missing_values_table(data, na_name=True)

    # 4. fix missing values - fill with median values
    if nan_cols:
        if verbose:
            print(f'\nFixing missing values...')
        data.loc[:, nan_cols] = data.loc[:, nan_cols].fillna(data.loc[:, nan_cols].median())

    if verbose:
        nan_cols = data.columns[data.isnull().any()].to_list()
        print(f' Checking columns with nulls: {nan_cols}')

    if REMOVE_OUTLIERS:
        outlier_cols = ['avg_class_frequency_current_month']
        for col in outlier_cols:
            if col in columns:
                print(f"\nRemoving {col} outliers using IQR")
                lower, upper = determine_outlier_thresholds_iqr(data, col, th1=0.1, th3=0.9)
                print(" upper limit:", upper)
                print(" lower limit:", lower)
                data = data[(data[col] >= lower) & (data[col] <= upper)]

    print(
        f'\nFinal number of records: {data.shape[0]} / {total_rows_number} =',
        f'{data.shape[0]/total_rows_number*100:05.2f}%\n',
    )
    return data


def enc_save(enc, file_name):
    with open(file_name, 'wb') as f:
        pickle.dump(enc, f)


def enc_load(file_name):
    with open(file_name, 'rb') as f:
        enc = pickle.load(f)
        return enc
    return OrdinalEncoder()


def preprocess_data(df, ord_enc, fit_enc=False, verbose=DEBUG):
    # fix missing values, remove outliers
    df = preprocess_df(df, verbose)

    # encode categorical
    categorical_features = df.select_dtypes(exclude=[np.number]).columns
    if len(categorical_features):
        if verbose:
            print('OrdinalEncoder categorical_features:', list(categorical_features))
        # import ordinal encoder from sklearn
        # ord_enc = OrdinalEncoder()
        if fit_enc:
            # Fit and Transform the data
            df[categorical_features] = ord_enc.fit_transform(df[categorical_features])
            enc_save(ord_enc, f'{MODEL_DIR}encoder.pkl')
            if verbose:
                print(' OrdinalEncoder categories:', ord_enc.categories_)
        else:
            # Only Transform the data (using pretrained encoder)
            df[categorical_features] = ord_enc.transform(df[categorical_features])

    columns = df.columns.to_list()
    if verbose and TARGET in columns and len(df) > 10:
        corr = df.corr(numeric_only=True)[TARGET]
        print(f'\nUpdated Correlation to {TARGET}:\n{corr.sort_values().to_string()}')

    return df


def load_data(verbose=DEBUG):
    df = load_dataset(verbose)
    ord_enc = OrdinalEncoder()
    df = preprocess_data(df, ord_enc, fit_enc=True, verbose=verbose)
    return df


if __name__ == '__main__':
    # quick test
    df = load_data()  # load dataset and train encoder
