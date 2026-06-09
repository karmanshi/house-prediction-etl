from datetime import datetime

from core.extract import data_extract
from core.transform import data_transform
from core.load import df_tocsv


input_file_path = r'./input_data/Dataset.csv'

if __name__ == "__main__":

    start_time=datetime.now()

    # creating a dataframe from file path
    extracted_dataframe = data_extract(input_file_path)
    print("Data Loading Successfull")

    # applying the necessary transformation on dataframe
    transformed_df = data_transform(extracted_dataframe)
    print("Data Transformation successfull")

    # exporting to csv
    file_path = df_tocsv(transformed_df)
    print("Data successfully exported, at output_data/cleaned_house_data.csv")

    end_time=datetime.now()
    print("Execution Time Taken: ", round((end_time-start_time).total_seconds()/60, 5), " minutes")