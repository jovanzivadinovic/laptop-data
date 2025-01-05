import seaborn as sns
import matplotlib.pyplot as plt
from modules.data_clean import CleanData
from modules.exp_data_analysis import ExpDataAnalysis
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics.pairwise import cosine_similarity

class Helper:
    def plot_price_trends(processor_price_trend, gpu_price_trend, ram_price_trend):
        sns.barplot(x=processor_price_trend.index, y=processor_price_trend.values)
        plt.title('Processor Price Trend')
        plt.xlabel('CPU')
        plt.ylabel('Average Price (€)')
        plt.xticks(rotation=45)  
        plt.tight_layout()
        plt.show()

        sns.barplot(x=gpu_price_trend.index, y=gpu_price_trend.values)
        plt.title('GPU Price Trend')
        plt.xlabel('GPU')
        plt.ylabel('Average Price (€)')
        plt.xticks(rotation=45) 
        plt.tight_layout()
        plt.show()

        sns.barplot(x=ram_price_trend.index, y=ram_price_trend.values)
        plt.title('RAM Price Trend')
        plt.xlabel('RAM')
        plt.ylabel('Average Price (€)')
        plt.xticks(rotation=45) 
        plt.tight_layout()
        plt.show()


    def cleanData(df):
        clean = CleanData()
        clean.formatResolution(df)
        clean.formatScreenSize(df)
        clean.seperate_memory_type(df)
        clean.clean_numeric_column(df, 'Memory', 'GB')
        clean.clean_numeric_column(df, 'Weight', 'kg')
        clean.clean_numeric_column(df, 'Ram', 'GB')
        clean.categorize_by_price(df)
        clean.seperate_cpu_type(df)
        clean.seperate_gpu_type(df)
        le_company = LabelEncoder()
        le_product = LabelEncoder()
        le_typeName = LabelEncoder()
        le_cpu = LabelEncoder()
        le_cpuModel = LabelEncoder()
        le_gpu = LabelEncoder()
        le_gpuModel = LabelEncoder()
        le_opsys = LabelEncoder()
        le_memoryType = LabelEncoder()

        clean.encode_column(le_company, df, "Company")
        clean.encode_column(le_product, df, "Product")
        clean.encode_column(le_typeName, df, "TypeName")
        clean.encode_column(le_cpu, df, "Cpu")
        clean.encode_column(le_cpuModel, df, "CpuModel")
        clean.encode_column(le_gpu, df, "Gpu")
        clean.encode_column(le_gpuModel, df, "GpuModel")
        clean.encode_column(le_opsys, df, "OpSys")
        clean.encode_column(le_memoryType, df, "MemoryType")

        return le_company, le_product, le_typeName, le_cpu, le_cpuModel, le_gpu, le_gpuModel, le_opsys, le_memoryType

    def get_similar_laptops_by_id(userVector, userPref, df, all_laptops_df):
        laptop_vectors = df[[col for col in userPref.keys() if col in df.columns]].values
        similarities = cosine_similarity(userVector, laptop_vectors)
        df['Similarity'] = similarities[0]
        df_sorted = df.sort_values(by='Similarity', ascending=False)
        top_3_ids = df_sorted.head(3)['laptop_ID']  
        similar_laptops = all_laptops_df[all_laptops_df['laptop_ID'].isin(top_3_ids)]
        return similar_laptops


    def EDA(df):
        expData = ExpDataAnalysis(df)
        print(expData.getMissingValues())
        expData.getBrandDistribution()
        expData.getPriceDistribution()
        expData.getScreenSizeDistribution()

        clean = CleanData()
        clean.seperate_cpu_type(df)
        expData.getCpuDistribution()
        expData.getRamDistribution()
        expData.getCompanyPriceRelation()
        expData.getScreenSizePriceRelation()
        expData.getCpuPriceRelation()
        expData.getRamPriceRelation()

        clean = CleanData()
        clean.seperate_memory_type(df)
        clean.formatScreenSize(df)
        clean.clean_numeric_column(df, 'Memory', 'GB')
        clean.clean_numeric_column(df, 'Ram', 'GB')
        corrData = ExpDataAnalysis(df)
        print(corrData.getStatInsights())
        corrData.getCorrelationMatrix()