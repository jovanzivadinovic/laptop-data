import pandas as pd

class CleanData:
    def clean_numeric_column(df, column_name, unit):
        df[column_name] = df[column_name].str.replace(unit, '')
        df[column_name] = pd.to_numeric(df[column_name])

    def formatResolution(df):
        df['Resolution'] = df['ScreenResolution'].str.extract(r'(\d+x\d+)')
        df[['Resolution_Width', 'Resolution_Height']] = df['Resolution'].str.split('x', expand=True).astype(int)
        df.drop('ScreenResolution', axis=1, inplace=True)
        df.drop('Resolution', axis=1, inplace=True)
        return df


    def formatScreenSize(df):
        df['Inches'] = df['Inches'].astype(float)
    
    def seperate_memory_type(df):
        df[['Memory', 'MemoryType']] = df['Memory'].str.split(' ', n=1, expand=True)
        if df['Memory'].str.contains('TB').any():
            df['Memory'] = df['Memory'].str.replace('TB', '000GB')

    def categorize_by_price(df):
        df['PriceRange'] = pd.qcut(df['Price_euros'], q=3, labels=False)
    
    def turn_to_categorical(df, column_name):
        df[column_name] = pd.factorize(df[column_name])[0]