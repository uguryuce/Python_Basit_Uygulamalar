# Gerekli kütüphane tanımlamaları ------------------------------------------------------------------
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

# Görselleştirme için import edildi
from matplotlib.colors import ListedColormap

# Standardization yapabilmek için gerekli
from sklearn.preprocessing import StandardScaler 

# KNN en iyi parametreleri bulurken kullanmak üzere import ettik
from sklearn.model_selection import train_test_split, GridSearchCV

# Test sonucundaki hataları bulmak için kullanılır | Hata Matrisi
from sklearn.metrics import accuracy_score, confusion_matrix

# Outlier detect işlemi için gerekli
from sklearn.neighbors import KNeighborsClassifier, NeighborhoodComponentsAnalysis, LocalOutlierFactor

# import PCA
from sklearn.decomposition import PCA

# uyarı kütüphanesi
import warnings
warnings.filterwarnings("ignore")








# Veri seti yükleme --------------------------------------------------------------------------------
data = pd.read_csv("column_2C.csv")








# class feature'ın isminin sinif olarak değiştirilmesi -----------------------------------------------
data = data.rename(columns = {"class":"sinif"})








# scatter ile iki farklı sınıf olarak görselleştirme -------------------------------------------------
A = data[data.sinif=='Abnormal']
N = data[data.sinif=='Normal']

plt.scatter(A.sacral_slope, A.degree_spondylolisthesis, color="blue", label="Abnormal")
plt.scatter(N.sacral_slope, N.degree_spondylolisthesis, color="red", label="Normal")
plt.xlabel("Sonuç")
plt.ylabel("Parametre")
plt.legend()

plt.scatter(A.pelvic_tilt_numeric, A.lumbar_lordosis_angle, color="black", label="Abnormal")
plt.scatter(N.pelvic_tilt_numeric, N.lumbar_lordosis_angle, color="yellow", label="Normal")
plt.xlabel("Sonuç")
plt.ylabel("Parametre")
plt.legend()








# sinif feature'ınde bulunan string değerlerin (Abnormal-Normal | 1-0) olarak değiştirilmesi ----------
def renameA(x):
    if x== 'Abnormal':
        return 1
    if x== 'Normal':
        return 0
    
data['sinif'] = data['sinif'].apply(renameA)








# Korelasyon işleminin gerçekleştirilmesi ve görselleştirilmesi --------------------------------------
corr_matrix = data.corr()
sns.clustermap(corr_matrix, annot = True, fmt = ".2f")
plt.title("Correlation Between Features")
plt.show()








# Box Plot işlemi gerçekleştirildi -------------------------------------------------------------------

# iki farklı class olduğu için iki farklı class şeklinde görselleştireceğiz ve data melt edilir
data_melted = pd.melt(data, id_vars = "sinif",
                      var_name = "features",
                      value_name = "value")

# yeni bir figür oluşturduk
plt.figure() 

# x-feature , y-value ve target diyerek classlara göre ayırdık
sns.boxplot(x = "features", y = "value", hue = "sinif", data = data_melted)

# feature isimleri 90 derece döndürülmüş şekilde verilecek
plt.xticks(rotation = 90)
plt.show()























