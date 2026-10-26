
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans

# Step 1: Create dataset
data = {
    "income": [20, 30, 40, 50, 60],
    "spend": [20, 30, 40, 50, 60]
}

# Step 2: Convert into DataFrame
df = pd.DataFrame(data)
print(df)

# Step 3: Select features
X = df[["income", "spend"]]

# Step 4: Create K-Means model
k = KMeans(
    n_clusters=2,
    random_state=42,
    n_init=10
)

# Step 5: Perform clustering
df["Cluster"] = k.fit_predict(X)

# Step 6: Display result
print("\nClustered Data:")
print(df)

# Step 7: Plot graph
plt.scatter(df["income"], df["spend"],
            c=df["Cluster"])
plt.xlabel("Income")
plt.ylabel("Spending")
plt.title("K-Means Clustering")
plt.show()
