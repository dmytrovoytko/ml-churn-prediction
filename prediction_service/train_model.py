# import warnings # supress warnings
# warnings.filterwarnings('ignore')

from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, balanced_accuracy_score, roc_auc_score
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.tree import DecisionTreeClassifier

import os
import pickle

from predict import print_results

from settings import DEBUG, MODEL_DIR, MODEL_NAME, TARGET # isort:skip
DEBUG = True # True # False # override global settings

SAVE_MODEL = True

def prepare_training(df, params):
    # dataset in df is already preprocessed - cleaned, OrdinalEncoder applied
    # determine test_size
    test_size = params.get('test_size', 0.2)
    # target labels column
    target = TARGET
    cols = df.columns.to_list()

    random_state = params.get('random_state', 42)
    cols.remove(target)
    X_train, X_test, y_train, y_test = train_test_split(df[cols], df[target], 
                                                        test_size=test_size, random_state=random_state)
    return X_train, X_test, y_train, y_test, params


def train_model(df, params, random_state=42):

        X_train, X_test, y_train, y_test, _ = prepare_training(df, {
                                                            'test_size': params['test_size'],
                                                            'random_state': random_state
                                                        })
        classifier_name = params['classifier']
        print(f"\n>>>>> Starting GridSearchCV {classifier_name}...")
        # Define the parameter grid to tune the hyperparameters
        if classifier_name=='AdaBoostClassifier':
            param_grid = {
                'n_estimators': [25, 50, 100, 125, 150],
                'learning_rate': [0.2, 0.6, 1.0, 1.4, 1.8], 
            }
            classifier = AdaBoostClassifier(algorithm='SAMME', random_state=random_state)
        elif classifier_name=='LogisticRegression':
            param_grid = {
                'C': [0.4, 0.6, 0.8, 1.0, 1.2, 1.4],
                'max_iter': [50, 100, 200, 300, 400],
            }
            classifier = LogisticRegression(solver='liblinear', random_state=random_state)
        elif classifier_name=='RandomForestClassifier':
            param_grid = {
                'n_estimators': [25, 50, 100, 150, 200],
                'max_depth': [3, 6, 9, None],
            }
            classifier = RandomForestClassifier(random_state=random_state)
        elif classifier_name=='DecisionTreeClassifier':
            param_grid = {
                'max_depth': [3, 5, 7, 10, 20, 30, None],
                'min_samples_leaf': [1, 3, 5],
                'min_samples_split': [2, 6, 10],
            }
            classifier = DecisionTreeClassifier(random_state=random_state)
        else:
            print('!! Unexpected classifier:', classifier_name)
            return

        grid_search = GridSearchCV(estimator=classifier, param_grid=param_grid,
                                   cv=params['cv'],
                                   n_jobs=-1,
                                   verbose=1, #3, #2,
                                   scoring=params['estimator'] # balanced_accuracy accuracy neg_mean_squared_error roc_auc_ovr
                                   )
        grid_search.fit(X_train, y_train)
        best_classifier = grid_search.best_estimator_ # Get the best estimator from the grid search
        best_params = grid_search.best_params_
        best_score = grid_search.best_score_
        print(f'\nHPO grid search: {grid_search.scorer_}, {grid_search.n_splits_} splits')
        print(f"Best parameters: {best_params}")

        # calculating test prediction metrics
        y_pred = best_classifier.predict(X_test)
        print_results(f"{classifier_name}, optimized for {params['estimator']}", y_test, y_pred, verbose=True)
        key_metric1 = accuracy_score(y_test, y_pred)
        key_metric2 = balanced_accuracy_score(y_test, y_pred)
        print(f" {params['estimator']} best_score_: {best_score:.3f}")
        if DEBUG:
            features_ = list(df.columns)
            features_.remove(TARGET)
            if classifier_name!='LogisticRegression':
                feature_importances_ = dict(zip(features_, list(grid_search.best_estimator_.feature_importances_)))
                print(classifier_name, 'feature_importances_', sorted(feature_importances_.items(), key=lambda x:x[1], reverse=True))

        if SAVE_MODEL:
            os.makedirs(MODEL_DIR, exist_ok=True)
            filename = f'{MODEL_DIR}{classifier_name}.pkl'
            pickle.dump(best_classifier, open(filename, 'wb'))
        return best_classifier, best_params, key_metric1, key_metric2


if __name__ == '__main__':
    from time import time

    from preprocess import load_data
    df = load_data()
    estimator = 'roc_auc_ovr' # 'accuracy'
    for classifier in [
                    'LogisticRegression',
                    'DecisionTreeClassifier',
                    'RandomForestClassifier',
                    'AdaBoostClassifier',
                    ]:
        t_start = time()
        params = {'classifier': classifier,
                    'estimator': estimator,
                    'cv': 2,
                    'test_size': 0.2,
                    'balance': False,
                    }
        best_classifier, best_params, key_metric1, key_metric2 = train_model(df, params, random_state=42)
        print(f' Finished in {(time() - t_start):.3f} second(s)\n')
