# Scoutium Yetenek Sınıflandırma

Futbolcuların scout (gözlemci) değerlendirmelerine dayanarak oyuncuların sınıfını (**average** / **highlighted**) tahmin eden bir makine öğrenmesi projesi.

## Problem

Scout'ların maçlarda oyuncuların çeşitli özelliklerine verdiği puanlardan yola çıkarak, bir oyuncunun "ortalama" mı yoksa "öne çıkan" bir yetenek mi olduğunu sınıflandırmak.

## Veri Seti

Proje iki CSV dosyasını kullanır (`SCOUTIUM/` klasörü altında, `;` ayraçlı):

| Dosya | İçerik |
|-------|--------|
| `scoutium_attributes.csv` | Oyuncu, maç, scout ve özellik bazında verilen puanlar |
| `scoutium_potential_labels.csv` | Oyunculara scout'lar tarafından atanan potansiyel etiketleri |

İki dosya `task_response_id`, `match_id`, `evaluator_id` ve `player_id` üzerinden birleştirilir.

## İşlem Adımları

1. **Veri birleştirme** — İki tablo `left join` ile birleştirilir (10730 satır, 9 değişken).
2. **Filtreleme**
   - Kaleci pozisyonu (`position_id == 1`) veriden çıkarılır (700 satır).
   - `below_average` etiketli satırlar çıkarılır (136 satır).
3. **Pivot** — Her oyuncu tek satır olacak şekilde, `attribute_id` değerleri sütunlara dönüştürülür (`attribute_value` ortalaması alınır).
4. **Encoding & Scaling**
   - `potential_label` → `LabelEncoder` ile binary kolona çevrilir.
   - Sayısal özellikler → `StandardScaler` ile ölçeklenir.
   - `position_id` → `OneHotEncoder` (`get_dummies`) ile kategorik olarak kodlanır.
5. **Modelleme** — Birden fazla algoritma 3 katlı cross-validation ile karşılaştırılır.
6. **Hiperparametre optimizasyonu** — En iyi modeller arasından Random Forest `GridSearchCV` ile optimize edilir.
7. **Değişken önemi** — Final modelde özelliklerin etkisi görselleştirilir.

## Model Sonuçları (Cross-Validation)

| Model | ROC AUC | F1 | Precision | Recall | Accuracy |
|-------|---------|-----|-----------|--------|----------|
| Logistic Regression | 0.8175 | 0.5787 | 0.7222 | 0.4990 | 0.8560 |
| **Random Forest** | **0.9035** | **0.6325** | **0.9333** | 0.4815 | **0.8855** |
| GBM | 0.8417 | 0.5796 | 0.6702 | 0.5351 | 0.8378 |
| XGBoost | 0.8563 | 0.5985 | 0.6799 | 0.5517 | 0.8488 |
| LightGBM | 0.8725 | 0.5848 | 0.6800 | 0.5175 | 0.8487 |

En iyi 3 model: **Random Forest**, **LightGBM**, **GBM**.

### Optimize Edilmiş Random Forest

En iyi parametreler:

```python
{'max_depth': 8, 'max_features': 10, 'min_samples_split': 5, 'n_estimators': 200}
```

ROC AUC skoru: **0.9009 → 0.9195** (one-hot pozisyon kolonları eklendikten sonra).

## Gereksinimler

```bash
pip install numpy pandas seaborn matplotlib scikit-learn xgboost lightgbm catboost
```

## Çalıştırma

```bash
python scoutiumsiniflandirma.py
```

> **Not:** Script, `SCOUTIUM/` klasöründeki iki CSV dosyasını bekler. Betiği çalıştırmadan önce dosyaların doğru yolda olduğundan emin olun.

## Dizin Yapısı

```
.
├── scoutiumsiniflandirma.py
├── README.md
└── SCOUTIUM/
    ├── scoutium_attributes.csv
    └── scoutium_potential_labels.csv
```
