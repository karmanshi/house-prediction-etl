
def df_tocsv(df, output_path='./output_data/cleaned_house_data.csv'):
    df.to_csv(output_path)
    return output_path
