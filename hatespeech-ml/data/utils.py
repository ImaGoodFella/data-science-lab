import pandas as pd

def clean_data(data_file, label_column, comment_column):

    df = pd.read_csv(data_file, low_memory=False)
    df_processed = df[~df[comment_column].isna()]

    df_processed = df_processed[[label_column, comment_column]]
    df_processed = df_processed.dropna()
    df_processed = df_processed.drop_duplicates(subset=comment_column)

    df_processed[label_column] = df_processed[label_column].astype(int)
    df_processed = df_processed.rename(mapper={comment_column : 'text', label_column : 'label'}, axis=1)
    #df_processed = df_processed.sample(256)

    return df_processed