import pandas as pd
import numpy as np
import re

def extract_storage_size(memory):
    ssd_size = re.search(r'(\d+)(?:GB|TB) SSD', memory)
    hdd_size = re.search(r'(\d+)(?:GB|TB) HDD', memory)
    hybrid_size = re.search(r'(\d+(?:\.\d+)?)TB Hybrid', memory)
    flash_size = re.search(r'(\d+)GB Flash Storage', memory)

    ssd_size_gb = int(float(ssd_size.group(1)) * 1024) if ssd_size and 'TB' in ssd_size.group() else int(ssd_size.group(1)) if ssd_size else 0
    hdd_size_gb = int(hdd_size.group(1)) * 1024 if hdd_size and 'TB' in hdd_size.group() else int(hdd_size.group(1)) if hdd_size else 0
    hybrid_size_gb = int(float(hybrid_size.group(1)) * 1024) if hybrid_size else 0
    flash_size_gb = int(flash_size.group(1)) if flash_size else 0

    return ssd_size_gb, hdd_size_gb, hybrid_size_gb, flash_size_gb

def cat_OS(inp):
    if inp in {'Windows 10', 'Windows 7', 'Windows 10 S'}:
        return 'Windows'
    elif inp in {'Mac OS X', 'macOS'}:
        return 'Mac'
    else:
        return 'Others/No OS/Linux'

def fetch_processor(data):
    if data in {'Intel Core i7','Intel Core i5', 'Intel Core i3'}:
        return data
    elif 'Intel' in data:
        return 'Other Intel Processor'
    return 'AMD Processor'

def load_data(filepath):
    df = pd.read_csv(filepath)
    return df

def transform_data(df):

    # print(df.duplicated().sum())
    # print(df.isnull().sum())

    df.drop(columns=['Unnamed: 0'], inplace= True)

    df['Ram'] = df['Ram'].str.replace('GB', '')
    df['Weight'] = df['Weight'].str.replace('kg', '')

    df['Ram'] = df['Ram'].astype('int32')
    df['Weight'] = df['Weight'].astype('float32')
    
    # zaokroaglenie ceny do typu calkowitego
    df['Price'] = df['Price'].round().astype(int)

    df['Touchscreen'] = df['ScreenResolution'].apply(lambda x:1 if 'Touchscreen' in x else 0)
    df['IPS'] = df['ScreenResolution'].apply(lambda x:1 if 'IPS' in x else 0)
    df['x_resolution'] = df['ScreenResolution'].apply(lambda x:x[x.index('x') - 4 :  x.index('x')])
    df['y_resolution'] = df['ScreenResolution'].apply(lambda x:x.split('x')[-1])
    df['x_resolution'] = df['x_resolution'].astype('int')
    df['y_resolution'] = df['y_resolution'].astype('int')
    df['ppi'] = (pow(pow(df['x_resolution'], 2) + pow(df['y_resolution'], 2), 0.5) /df['Inches'] ).astype(float)
    # print(df.sample(n=5))

    # usuniecie ponizszych column ze wzgledu na korelacje, zastapi je kolumna ppi
    df.drop(columns=['x_resolution', 'y_resolution', 'ScreenResolution', 'Inches'], inplace=True)

    df['Cpu Name'] = df['Cpu'].apply(lambda x:' '.join(x.split()[ : 3]))
    df['Cpu Brand'] = df['Cpu Name'].apply(fetch_processor)
    df.drop(columns=['Cpu Name', 'Cpu'], inplace=True)

    df['SSD'], df['HDD'], df['Hybrid'], df['Flash Storage'] = zip(*df['Memory'].apply(extract_storage_size))
    df.drop(columns=['Memory'], inplace=True)
    df.drop(columns=['Hybrid', 'Flash Storage'], inplace=True)

    df['Gpu Brand'] = df['Gpu'].apply(lambda x: x.split()[0])
    #usuniecie wierszy z GPU Brand = ARM -> jest tylko 1 taki rekord
    df = df[df['Gpu Brand'] != 'ARM']
    df.drop(columns=['Gpu'], inplace= True)

    df['OS'] = df['OpSys'].apply(cat_OS)
    df.drop(columns=['OpSys'], inplace=True)
    # print(df.head())
    # print(df.info())

    X = df.drop(columns=['Price'])
    y = np.log(df['Price'])
    # print(X.head())
    # print(y.head())
    
    return df

def save_data(df, filepath):
    df.to_csv(filepath, index=False)

def run_etl_process(input_filepath, output_filepath):
    df = load_data(input_filepath)
    df_transformed = transform_data(df)
    save_data(df_transformed, output_filepath)
    
    print("ETL process completed successfully.")

input_filepath = 'laptop_data_input_file.csv'
output_filepath = 'laptop_data_output_file.csv'
run_etl_process(input_filepath, output_filepath)



