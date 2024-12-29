import pandas as pd
from mlxtend.frequent_patterns import apriori, association_rules
from mlxtend.preprocessing import TransactionEncoder

# 1. CSV Dosyasını Okuma
dataset = pd.read_csv("Updated_Electric_Vehicle_Data_VIN.csv")

# 2. İlgili Sütunları Seçme ve Birleştirme
selected_columns = [
    'County', 'City', 'State', 'Model Year', 'Make', 
    'Model', 'Electric Vehicle Type', 'Clean Alternative Fuel Vehicle CAFV Eligibility'
]
basket = dataset[selected_columns]

# 3. Veriyi Dönüştürme ve Filtreleme
transactions = (
    basket.dropna()          # Eksik verileri kaldır
    .applymap(str)           # Tüm değerleri stringe çevir
    .apply(list, axis=1)     # Her satırı listeye çevir
    .tolist()                # Tüm dataframe'i listeye dönüştür
)

# 4. TransactionEncoder ile İşleme
te = TransactionEncoder()
te_ary = te.fit_transform(transactions)
df = pd.DataFrame(te_ary, columns=te.columns_)

# 5. Apriori Algoritması ile Sık Geçen Öğe Gruplarını Bulma
min_support = 0.2  # Minimum destek değeri
frequent_itemsets = apriori(df, min_support=min_support, use_colnames=True)

if frequent_itemsets.empty:
    print(f"No frequent itemsets found for minimum support {min_support}.")
else:
    print(f"Frequent itemsets for minimum support {min_support}:")
    print(frequent_itemsets)
    # Sonuçları CSV'ye kaydet
    frequent_itemsets.to_csv("frequent_itemsets.csv", index=False)
    print("Frequent itemsets saved to 'frequent_itemsets.csv'.")

# 6. Birliktelik Kurallarını Hesaplama
if not frequent_itemsets.empty:
    rules = association_rules(
        frequent_itemsets, 
        metric="confidence", 
        min_threshold=0.7, 
        num_itemsets=df.shape[0]
    )
    
    if rules.empty:
        print("No association rules generated.")
    else:
        # Kuralları güven değerine göre sırala
        rules = rules.sort_values(by="confidence", ascending=False)
        print("Top Association Rules:")
        print(rules[['antecedents', 'consequents', 'support', 'confidence', 'lift']])
        
        # Sonuçları CSV'ye kaydet
        rules.to_csv("association_rules.csv", index=False)
        print("Association rules saved to 'association_rules.csv'.")
