import requests
from settings import DATA_DIR, MODEL_DIR, PORT, TARGET

DEBUG = True # True # False # override global settings

if __name__ == '__main__':
    # quick tests
    print('\nTesting predict_df...')
    # batch testing with 10 records 
    from preprocess import load_data

    df = load_data(verbose=False).head(10)

    url = f'http://localhost:{PORT}/predict'
    print('Testing web service', url)
    for row in df.to_dict('records'):
        try:
            response = requests.post(url, json=row)
            print('\n data:', row)
            print(' response:', response.status_code)
            if response.status_code==200:
                print('   source:', row[TARGET], ' -> prediction:', response.json())
            else:
                print('   error:', response.text)
        except Exception as e:
            print('   error:', e)

