import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity

# Load dataset (change file path if needed)
data = pd.read_excel("data.xlsx")

# Data preprocessing
data.dropna(inplace=True)
data['CustomerID'] = data['CustomerID'].astype(int)

# Create user-item matrix
user_item = data.pivot_table(index='CustomerID', columns='StockCode', values='Quantity', fill_value=0)

# Compute similarity
similarity = cosine_similarity(user_item)

# Convert to DataFrame
similarity_df = pd.DataFrame(similarity, index=user_item.index, columns=user_item.index)

# Recommendation function
def recommend_products(customer_id, top_n=5):
    similar_users = similarity_df[customer_id].sort_values(ascending=False)[1:6]
    similar_users_ids = similar_users.index

    recommended_items = user_item.loc[similar_users_ids].sum().sort_values(ascending=False)

    print("\nRecommended products:")
    print(recommended_items.head(top_n))

# Run example
if __name__ == "__main__":
    print("Available Customer IDs:", list(user_item.index[:5]))
    cid = int(input("Enter Customer ID: "))
    recommend_products(cid)
