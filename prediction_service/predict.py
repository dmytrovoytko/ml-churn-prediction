import pickle

import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    classification_report,
    confusion_matrix,
)

from preprocess import enc_load, preprocess_data
from settings import DATA_DIR, MODEL_DIR, MODEL_NAME, TARGET, USE_ENCODER

from settings import DEBUG  # isort:skip

DEBUG = True # True # False # override global settings

def print_results(classifier, y_test, predict_y, verbose=DEBUG):
    print(f'\n================\n{classifier}')
    if verbose:
        print(classification_report(y_test, predict_y, digits=3))
        print(confusion_matrix(y_test, predict_y))
    print(f' accuracy: {accuracy_score(y_test, predict_y):05.3f}')
    print(f' balanced accuracy: {balanced_accuracy_score(y_test, predict_y):05.3f}')


def predict_df(df, MODEL_DIR, verbose=DEBUG):
    print(f'\nPredicting using model {MODEL_DIR}{MODEL_NAME}')

    try:
        test_data = preprocess_data(df)
    except:
        test_data = df

    cols = test_data.columns.to_list()
    X_test = pd.DataFrame(test_data, columns=cols)
    if TARGET in cols:
        y_test = X_test.pop(TARGET)
    else:
        y_test = pd.Series()

    try:
        model = pickle.load(open(f'{MODEL_DIR}{MODEL_NAME}', 'rb'))
        if DEBUG:
            print(' model.get_params:', model.get_params())
    except Exception as e:
        print('!!! Exception while loading model:', e)
        return pd.Series()

    try:
        y_pred = model.predict(X_test)
        if verbose:
            print(f"\nPrediction:\ny_pred: {list(y_pred)[:10]}")
            if len(y_test):
                print(f"y_test: {list(y_test)[:10]}")
        return y_pred
    except Exception as e:
        print(f'!!! Exception while predicting {TARGET}:', e)
        return pd.Series()


def predict_dict(object, verbose=False):
    # transform dict to create df from it
    dct = {k: [v] for k, v in object.items()}
    df = pd.DataFrame(dct)
    if USE_ENCODER:
        ord_enc = enc_load(f'{MODEL_DIR}encoder.pkl')
    else:
        ord_enc = None
    df = preprocess_data(df, ord_enc, fit_enc=False)
    y_pred = predict_df(df, MODEL_DIR, verbose=verbose)
    result = {
        str(TARGET).lower(): int(list(y_pred)[0])
    }  # explicit int() is reqiered for serialization
    if DEBUG:
        print('result:', result)

    return result


if __name__ == '__main__':
    # quick tests
    print('\nTesting predict_df...')
    # batch testing with 10 records 
    from preprocess import load_data

    df = load_data().head(10)
    y_pred = predict_df(df, MODEL_DIR, verbose=True)

    columns = df.columns.to_list()
    if TARGET in columns:
        print_results('Trained model', df[TARGET], y_pred, verbose=True)

    # testing with 1 record 
    print('\n------------\n\nTesting predict_dict...')
    row = df.head(1).to_dict('records')[0]
    expected_result = row.pop(TARGET, None)
    result = predict_dict(row, verbose=True)
    print(f'\npredict_dict:\n data: {row}\n -> {result} // expected: {expected_result}')
