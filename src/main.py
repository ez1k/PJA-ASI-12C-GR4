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
    print(X.head())
    print(y.head())
    
    return df
def transform_data2(laptop_data):
    laptop_data.drop('Unnamed: 0', axis=1, inplace=True)
    laptop_data['Ram'] = laptop_data['Ram'].str.replace('GB', '').astype(int)
    laptop_data['Weight'] = laptop_data['Weight'].str.replace('kg', '').astype(float)
    # features
    # screen
    is_fullhd = laptop_data['ScreenResolution'].str.contains('Full HD')
    laptop_data['IsFullHD'] = is_fullhd.astype(int)

    is_ips = laptop_data['ScreenResolution'].str.contains('IPS')
    laptop_data['IsIPS'] = is_ips.astype(int)

    is_touchscreen = laptop_data['ScreenResolution'].str.contains('Touchscreen')
    laptop_data['IsTouchscreen'] = is_touchscreen.astype(int)

    is_retina = laptop_data['ScreenResolution'].str.contains('Retina')
    laptop_data['IsRetina'] = is_retina.astype(int)

    resolution = laptop_data['ScreenResolution'].str.extract(r'(\d+)x(\d+)')
    laptop_data['ResolutionWidth'] = resolution[0].astype(int)
    laptop_data['ResolutionHeight'] = resolution[1].astype(int)

    # memory
    is_hdd = laptop_data['Memory'].str.contains('HDD')
    laptop_data['IsHDD'] = is_hdd.astype(int)

    is_ssd = laptop_data['Memory'].str.contains('SSD')
    laptop_data['IsSSD'] = is_ssd.astype(int)

    is_flashstorage = laptop_data['Memory'].str.contains('Flash Storage')
    laptop_data['IsFlashStorage'] = is_flashstorage.astype(int)

    memory_size_hdd = laptop_data['Memory'].str.extract(r'(\d+)TB HDD')
    memory_size_hdd_gb = laptop_data['Memory'].str.extract(r'(\d+)GB HDD')
    memory_size_ssd = laptop_data['Memory'].str.extract(r'(\d+)GB SSD')
    flash_storage = laptop_data['Memory'].str.extract(r'(\d+)GB Flash Storage')

    laptop_data['FlashStorage'] = flash_storage[0].astype(float)
    laptop_data['MemorySizeHDD_TB'] = memory_size_hdd[0].astype(float)
    laptop_data['MemorySizeHDD_GB'] = memory_size_hdd_gb[0].astype(float)
    laptop_data['MemorySizeSSD'] = memory_size_ssd[0].astype(float)

    # cpu
    cpu_speed = laptop_data['Cpu'].str.extract(r'(\d+\.\d+)GHz')
    cpu_speed = cpu_speed.fillna(laptop_data['Cpu'].str.extract(r'(\d+)GHz'))
    laptop_data['CpuSpeed'] = cpu_speed[0].astype(float)

    cpu_brand = laptop_data['Cpu'].str.extract(r'([A-Za-z]+)')
    laptop_data['CpuBrand'] = cpu_brand[0]

    # gpu
    gpu_brand = laptop_data['Gpu'].str.extract(r'([A-Za-z]+)')
    laptop_data['GpuBrand'] = gpu_brand[0]

    # operating system
    laptop_data['OpSys'] = laptop_data['OpSys'].str.lower()
    laptop_data['OperatingSystem'] = laptop_data['OpSys'].apply(lambda x: 'macos' if 'mac' in x else (
        'windows' if 'windows' in x else ('linux' if 'linux' in x else 'freedos/other')))
    # drop some columns
    laptop_data.drop(['ScreenResolution', 'Memory', 'OpSys'], axis=1, inplace=True)
    print(laptop_data.info())
    return laptop_data
def save_data(df, filepath):
    df.to_csv(filepath, index=False)

def run_etl_process(input_filepath, output_filepath):
    # df = load_data(input_filepath)
    # df_transformed = transform_data(df)
    # save_data(df_transformed, output_filepath)

    df2 = load_data(input_filepath)
    df_transformed2 = transform_data2(df2)
    save_data(df_transformed2, output_filepath)
    
    print("ETL process completed successfully.")

input_filepath = 'laptop_data_input_file.csv'
output_filepath = 'laptop_data_output_file.csv'
run_etl_process(input_filepath, output_filepath)



