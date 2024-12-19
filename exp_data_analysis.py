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
        plt.show()

    def getBrandDistribution(self):
        sns.countplot(y='Company', data=self.df)
        plt.title('Brand Distribution')
        plt.show()

    def getRamDistribution(self):
        sns.countplot(y='Ram', data=self.df)
        plt.title('Ram Distribution')
        plt.show()

    def getCpuDistribution(self):
        sns.countplot(y='Cpu', data=self.df)
        plt.title('Cpu Distribution')
        plt.show()

    def getScreenSizeDistribution(self):
        sns.countplot(y='ScreenResolution', data=self.df)
        plt.title('Screen Size Distribution')
        plt.show()

    def getRamPriceRelation(self):
        sns.boxplot(x='Ram', y='Price_euros', data=self.df)
        plt.title('Ram-Price Relation')
        plt.show()

    def getCpuPriceRelation(self):
        sns.boxplot(x='Cpu', y='Price_euros', data=self.df)
        plt.title('Cpu-Price Relation')
        plt.show()

    def getScreenSizePriceRelation(self):
        sns.boxplot(x='ScreenResolution', y='Price_euros', data=self.df)
        plt.title('Screen Size-Price Relation')
        plt.show()

    def getCompanyPriceRelation(self):
        sns.boxplot(x='Company', y='Price_euros', data=self.df)
        plt.title('Brand-Price Relation')
        plt.show()

    def getCorrelationMatrix(self):
        plt.title('Correlation Matrix')
        return self.df.corr()

    def getStatInsights(self):
        plt.title('Statistical Insights')
        return self.df.describe()

