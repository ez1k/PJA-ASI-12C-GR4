import pandas as pd

def load_data(filepath):
    df = pd.read_csv(filepath)
    return df

def transform_data(df):
    df.fillna(0, inplace=True)
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

