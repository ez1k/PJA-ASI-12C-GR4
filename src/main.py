from ingestion import load_data
from transformation import transform_data
from ml import price_prediction_model

def run_etl_process(input_filepath, output_filepath):
    df = load_data.load_data(input_filepath)
    df_transformed = transform_data.transform_data(df)
    transform_data.save_data(df_transformed, output_filepath)
    print("ETL process completed successfully.")

def main():
    input_filepath = 'dataset/laptop_data_input_file.csv'
    output_filepath = 'dataset/laptop_data_output_file.csv'
    run_etl_process(input_filepath, output_filepath)
    price_prediction_model.run_price_prediction_model(output_filepath)

if __name__ == "__main__":
    main()



