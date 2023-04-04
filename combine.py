import pandas as pd


if __name__ == "__main__":
    # Read in the first dataframe
    df1 = pd.read_csv('new.csv')

    # Read in the second dataframe
    df2 = pd.read_csv('instagram_SA_profile.csv')

    # Extract the filename from the imgUrl column in df2
    df2['filename'] = df2['imgUrl'].apply(lambda x: x.split('?')[0].split('/')[-1])

    # Join the two dataframes based on filename and image_name
    merged_df = pd.merge(df1, df2, left_on='image_name', right_on='filename')

    # Drop the filename column, since it's no longer needed
    merged_df = merged_df.drop('filename', axis=1)

    # Write the merged dataframe to a new CSV file
    merged_df.to_csv('merged.csv', index=False)
