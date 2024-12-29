import pandas as pd
from mlxtend.frequent_patterns import apriori
from mlxtend.preprocessing import TransactionEncoder
from mlxtend.frequent_patterns import association_rules

# CSV dosyasını okuma (önceden düzenlenmiş olması gerekiyor)
dataset = pd.read_csv("Electric_Vehicle_Population_Data_One_Encoding2_forAL_13134.csv")

# print(dataset.columns)

# Sadece belirli sütunları seçme (örneğin, programlama dilleri)
basket = dataset[['Model', 'Electric Vehicle Type', 'City']]

print(basket['Model'].value_counts())
print(basket['Electric Vehicle Type'].value_counts())
print(basket['City'].value_counts())



# if 'Year' in dataset.columns:
#     basket = dataset[['Model', 'Year', 'Electric Vehicle Type', 'City']]
# else:
#     print("Year column not found. Using other columns for analysis.")
#     basket = dataset[['Model', 'Electric Vehicle Type', 'City']]
    
# TransactionEncoder ile dönüştürme
te = TransactionEncoder()
te_ary = te.fit(basket).transform(basket)
df = pd.DataFrame(te_ary, columns=te.columns_)

# Apriori algoritması
# Try different minimum support values
min_support_values = [0.05, 0.02, 0.01, 0.005]  # Experiment with different values


for min_support in min_support_values:
  frequent_itemsets = apriori(df, min_support=min_support, use_colnames=True)
  if not frequent_itemsets.empty:
    print(f"Frequent itemsets for minimum support {min_support}")
    print(frequent_itemsets)
    break  # Exit the loop if frequent itemsets are found

# # Birliktelik kuralları
# # Association rules oluşturma
# num_transactions = 13143
# rules = association_rules(frequent_itemsets, metric="confidence", min_threshold=0.7, num_itemsets=num_transactions)

# Sonuçları görüntüleme
print(frequent_itemsets)