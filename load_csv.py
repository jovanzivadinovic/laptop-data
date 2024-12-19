import pandas as pd


class LoadCsv:

    def load_csv(self, csv_path):
        df = pd.read_csv(csv_path, encoding='latin1')
        return df

