import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

class ExpDataAnalysis:
    def __init__(self, df):
        self.df = df

    def getMissingValues(self):
        return self.df.isnull().sum()

    def getPriceDistribution(self):
        sns.displot(self.df['Price_euros'])
        plt.title('Price Distribution')
        plt.tight_layout()
        plt.show()

    def getBrandDistribution(self):
        sns.countplot(y='Company', data=self.df)
        plt.title('Brand Distribution')
        plt.tight_layout()
        plt.show()

    def getRamDistribution(self):
        sns.countplot(y='Ram', data=self.df)
        plt.title('Ram Distribution')
        plt.tight_layout()
        plt.show()

    def getCpuDistribution(self):
        sns.countplot(y='Cpu', data=self.df)
        plt.title('Cpu Distribution')
        plt.xticks(rotation=90)
        plt.tight_layout()
        plt.show()

    def getScreenSizeDistribution(self):
        sns.countplot(y='ScreenResolution', data=self.df)
        plt.title('Screen Size Distribution')
        plt.tight_layout()
        plt.show()

    def getRamPriceRelation(self):
        sns.boxplot(x='Ram', y='Price_euros', data=self.df)
        plt.title('Ram-Price Relation')
        plt.tight_layout()
        plt.show()

    def getCpuPriceRelation(self):
        sns.boxplot(x='Cpu', y='Price_euros', data=self.df)
        plt.title('Cpu-Price Relation')
        plt.tight_layout()
        plt.show()

    def getScreenSizePriceRelation(self):
        sns.boxplot(x='ScreenResolution', y='Price_euros', data=self.df)
        plt.title('Screen Size-Price Relation')
        plt.tight_layout()
        plt.show()

    def getCompanyPriceRelation(self):
        sns.boxplot(x='Company', y='Price_euros', data=self.df)
        plt.title('Brand-Price Relation')
        plt.tight_layout()
        plt.show()

    def getCorrelationMatrix(self):
        # plt.title('Correlation Matrix')
        return self.df.corr()

    def getStatInsights(self):
        # plt.title('Statistical Insights')
        return self.df.describe()
    
    def brand_analysis(self):
        brand_analysis = self.df.groupby('Company').agg({
            'Price_euros': 'mean',
            'Ram': 'mean',
            'Inches': 'mean',
        }).reset_index()

        return brand_analysis



