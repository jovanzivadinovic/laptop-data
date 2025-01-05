import pandas as pd

class CleanData:
    def clean_numeric_column(self, df, column_name, unit):
        df[column_name] = df[column_name].str.replace(unit, '')
        df[column_name] = pd.to_numeric(df[column_name])
        return df

    def formatResolution(self, df):
        df['Resolution'] = df['ScreenResolution'].str.extract(r'(\d+x\d+)')
        df[['Resolution_Width', 'Resolution_Height']] = df['Resolution'].str.split('x', expand=True).astype(int)
        df.drop('ScreenResolution', axis=1, inplace=True)
        df.drop('Resolution', axis=1, inplace=True)
        return df


    def formatScreenSize(self, df):
        df['Inches'] = df['Inches'].astype(float)
        return df
    
    def seperate_memory_type(self, df):
        df[['Memory', 'MemoryType']] = df['Memory'].str.split(' ', n=1, expand=True)
        if df['Memory'].str.contains('TB').any():
            df['Memory'] = df['Memory'].str.replace('TB', '000GB')
        return df

    def categorize_by_price(self, df):
        df['PriceRange'] = pd.qcut(df['Price_euros'], q=3, labels=False)
        return df
    
    def encode_column(self, le, df, column_name):
        values = df[column_name].unique()
        le.fit(values)
        df[column_name] = le.transform(df[column_name])
        return df
    
    def seperate_cpu_type(self, df):
        df[['Cpu', 'CpuModel']] = df['Cpu'].str.split(' ', n=1, expand=True)
        return df
    
    def seperate_gpu_type(self, df):
        df[['Gpu', 'GpuModel']] = df['Gpu'].str.split(' ', n=1, expand=True)
        return df