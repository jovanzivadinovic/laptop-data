import pandas as pd

class CleanData:
    def __init__(self, df):
        self.df = df

    def clean_numeric_column(self, column_name, unit):
        self.df[column_name] = self.df[column_name].str.replace(unit, '')
        self.df[column_name] = pd.to_numeric(self.df[column_name])

    def formatResolution(self):
        self.df['Resolution'] = self.df['ScreenResolution'].str.extract(r'(\d+x\d+)')
        self.df[['Resolution_Width', 'Resolution_Height']] = self.df['Resolution'].str.split('x', expand=True).astype(int)
        self.df.drop('Resolution', axis=1, inplace=True)

    def formatScreenSize(self):
        self.df['Inches'] = self.df['Inches'].astype(float)

    def check_missing_values(self):
        return self.df.isnull().sum()