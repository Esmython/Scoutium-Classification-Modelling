import numpy as np
import pandas as pd
import seaborn as sns
from matplotlib import pyplot as plt
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier, VotingClassifier
from sklearn.model_selection import GridSearchCV, cross_validate, RandomizedSearchCV, validation_curve
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier

from xgboost import XGBClassifier
from lightgbm import LGBMClassifier
from catboost import CatBoostClassifier

pd.set_option('display.max_columns', None)
pd.set_option('display.width', 500)
import warnings
warnings.simplefilter(action='ignore', category=Warning)

# Dosyaları dataframe e aktarma
df1 = pd.read_csv("SCOUTIUM/scoutium_attributes.csv", sep=";")
df2 = pd.read_csv("SCOUTIUM/scoutium_potential_labels.csv", sep=";")
df1.head()
df2.head()

# Dataframe'leri birleştirme
df_ = pd.merge(df1, df2, how="left",
               on=["task_response_id", "match_id", "evaluator_id", "player_id"])
df = df_.copy()
df.head()
df.info() #10730 satır 9 değişken

# Kaleci pozisyonunu veriden çıkarma
df["position_id"].value_counts()
df = df[df["position_id"] != 1]
df["position_id"].value_counts() #700 satır silindi

# Ortalama altı etiketlileri veriden çıkarma
df["potential_label"].value_counts()
df = df[df["potential_label"] != "below_average"]
df["potential_label"].value_counts() #136 satır silindi

# Pivot ile attribute_id değerlerinden yeni satırlar oluşturma
pvt = pd.pivot_table(df,
                    index=["player_id", "position_id", "potential_label"],
                    columns="attribute_id",
                    values="attribute_value",
                    aggfunc="mean")

pvt = pvt.reset_index()
pvt.columns = [str(col) for col in pvt.columns]
#pvt.columns = pvt.columns.map(str)
pvt.head()
pvt.info()

# potential_label satırı lebal encoder ile binary kolona dönüştürme
def label_encoder(dataframe, binary_col):
    labelencoder = LabelEncoder()
    dataframe[binary_col] = labelencoder.fit_transform(dataframe[binary_col])
    return dataframe

label_encoder(pvt, "potential_label")

#df["potential_label"] = df["potential_label"].map({"average": 0, "highlighted": 1})

# Sayısal kolon listesi oluşturma
num_cols = pvt.columns.drop(["player_id", "position_id", "potential_label"]).tolist()

# Sayısal değerleri Standart Scaler ile position_id kolonunu ise One Hot Encoder ile ölçeklendirme
ss = StandardScaler()
pvt[num_cols] = ss.fit_transform(pvt[num_cols])
pvt.head(10)

pvt = pd.get_dummies(pvt, columns=["position_id"], drop_first=True)
pvt.head()
one_hot = [col for col in pvt.columns if "position" in col]
one_hot

# Modelleme
y = pvt["potential_label"]
X = pvt[num_cols + one_hot]

models = [
    ("LR", LogisticRegression()),
    # ("KNN", KNeighborsClassifier()),
    # ("SVC", SVC()),
    # ("CART", DecisionTreeClassifier()),
    ("RF", RandomForestClassifier()),
    # ("Adaboost", AdaBoostClassifier()),
    ("GBM", GradientBoostingClassifier()),
    ("XGBoost", XGBClassifier(eval_metric="logloss", verbosity=0)),
    # ("CatBoost", CatBoostClassifier(verbose=False)),
    ("LightGBM", LGBMClassifier(verbose=-1))
]

scoring = ["roc_auc", "f1", "precision", "recall", "accuracy"]

for name, model in models:
    cv_results = cross_validate(model, X, y, cv=3, scoring=scoring)
    print(f"########## {name} ##########")
    for metric in scoring:
        print(f"{metric}: {round(cv_results['test_' + metric].mean(), 4)}")

"""########## LR ##########
roc_auc: 0.8175
f1: 0.5787
precision: 0.7222
recall: 0.499
accuracy: 0.856
########## RF ##########
roc_auc: 0.9035
f1: 0.6325
precision: 0.9333
recall: 0.4815
accuracy: 0.8855
########## GBM ##########
roc_auc: 0.8417
f1: 0.5796
precision: 0.6702
recall: 0.5351
accuracy: 0.8378
########## XGBoost ##########
roc_auc: 0.8563
f1: 0.5985
precision: 0.6799
recall: 0.5517
accuracy: 0.8488
########## LightGBM ##########
roc_auc: 0.8725
f1: 0.5848
precision: 0.68
recall: 0.5175
accuracy: 0.8487
"""
""" Best3:  RF , LightGBM , GBM """

# Parametre optimizasyonu
rf_params = {"max_depth": [3, 5, 8, None],
             "max_features": [5, 10, "sqrt"],
             "min_samples_split": [2, 5, 10, 15],
             "n_estimators": [200, 500]}

rf_model = RandomForestClassifier(random_state=17)

rf_best_grid = GridSearchCV(rf_model, rf_params,
                            cv=3, n_jobs=-1, verbose=1,
                            scoring="roc_auc").fit(X, y)

print(rf_best_grid.best_params_)
# {'max_depth': 8, 'max_features': 10, 'min_samples_split': 5, 'n_estimators': 200}
# 'max_depth': Ağacın maksimum derinliği. Kaç kez üst üste dallanabileceğini sınırlar.
#              Yüksek/None = ağaç sonuna kadar büyür, aşırı öğrenme riski.
#              Düşük = daha basit, daha genellenebilir model.
#
# 'max_features': Her bölünmede rastgele seçilecek değişken sayısı.
#                 Düşük değer ağaçları birbirinden farklılaştırır,
#                 topluluğun (ensemble) gücünü artırır. Varsayılan "sqrt".

# 'min_samples_split': Bir düğümün bölünebilmesi için gereken minimum örnek sayısı.
#                      Varsayılan 2. Yükseltmek ağacın erken durmasını sağlar,
#                      az örnekli düğümlerde ezberlemeyi engeller.
#
# 'n_estimators': Fit edilecek bağımsız ağaç sayısı. Arttırmak genelde skoru
#                 iyileştirir veya sabitler, aşırı öğrenmeye yol açmaz —
#                 sadece eğitim süresini uzatır.

print(rf_best_grid.best_score_)
# One hot kolonları eklemeden önce 0.9009
# sonra: 0.9195

final_model = rf_best_grid.best_estimator_

# Değişkenleri sınıflandırmaya etkilerinin büyüklüklerine göre sıralamave bunu görselleştirme
def plot_importance(model, features, num=len(X), save=False):
    feature_imp = pd.DataFrame({'Value': model.feature_importances_, 'Feature': features.columns})
    plt.figure(figsize=(10, 10))
    sns.set(font_scale=1)
    sns.barplot(x="Value", y="Feature", data=feature_imp.sort_values(by="Value",
                                                                      ascending=False)[0:num])
    plt.title('Features')
    plt.tight_layout()
    plt.show()
    if save:
        plt.savefig('importances.png')

plot_importance(final_model, X)

# position_id_9: Ofansif orta saha
# position_id_6: Merkez orta saha


