KMeans Clustering
---

The determination of the optimal number of clusters in **K-Means** is a challenging problem because the algorithm requires the user to specify $( k )$, the number of clusters, in advance. There are various methods to estimate $( k )$ based on mathematical and statistical principles. Here are some of the most common techniques used:

### 1. **Elbow Method**

The **elbow method** is one of the most widely used techniques for determining the optimal number of clusters.

#### Steps:
- Run K-Means clustering for a range of $( k )$ values (e.g., $( k = 1 )$ to $( k = 10 )$).
- For each $( k )$, calculate the **Within-Cluster Sum of Squares (WCSS)** or **inertia**, which measures the compactness of the clusters. This is the sum of squared distances between each data point and its assigned cluster centroid.

The formula for WCSS for a single cluster $( j )$ is:
$$
\text{WCSS}_j = \sum_{i \in C_j} ||x_i - \mu_j||^2
$$
Where:
- $( x_i )$ is a data point in cluster $( C_j )$,
- $( \mu_j )$ is the centroid of cluster $( C_j )$,
- $( ||x_i - \mu_j||^2 )$ is the Euclidean distance between the data point and the cluster centroid.

Summing over all clusters gives the total WCSS:
$$
\text{WCSS}_{\text{total}} = \sum_{j=1}^{k} \text{WCSS}_j
$$

- Plot the WCSS against $( k )$ values. As $( k )$ increases, WCSS will decrease because the clusters become smaller and tighter. The idea is to find the point where adding more clusters no longer significantly improves the WCSS. This point is known as the **elbow**, and it indicates the optimal number of clusters.

#### Mathematical Interpretation:
When $( k )$ increases, the decrease in WCSS becomes less significant. The "elbow" point is where the rate of decrease sharply changes, indicating the point of diminishing returns.

### 2. **Silhouette Score**

The **silhouette score** measures how similar each point is to its own cluster (cohesion) compared to other clusters (separation).

For each point $( i )$, the silhouette coefficient $( s(i) )$ is defined as:
$$
s(i) = \frac{b(i) - a(i)}{\max(a(i), b(i))}
$$
Where:
- $( a(i) )$ is the average distance between point $( i )$ and all other points in its own cluster.
- $( b(i) )$ is the average distance between point $( i )$ and points in the nearest neighboring cluster.

- $( s(i) )$ ranges from -1 to 1. A value close to 1 means the point is well clustered, while a value near -1 indicates that the point is misclassified.

To find the optimal $( k )$, compute the silhouette score for different values of $( k )$ and choose the $( k )$ that maximizes the average silhouette score.

#### Mathematical Interpretation:
The silhouette score quantifies the separation between clusters. A higher silhouette score suggests that the clusters are well separated and that the points are well matched to their own cluster.

### 3. **Gap Statistic Method**

The **Gap Statistic** compares the WCSS of the K-Means solution with the expected WCSS under a null reference distribution (randomly distributed data points). This helps in understanding how much better the clustering is compared to random noise.

#### Steps:
- For each $( k )$, compute the WCSS for the clustering solution $( \text{WCSS}_k )$.
- Generate multiple random datasets (following a uniform distribution over the data’s bounding box) and compute the WCSS for these random datasets.
- The gap statistic is then defined as:
  $$
  \text{Gap}(k) = \frac{1}{B} \sum_{b=1}^{B} \log(\text{WCSS}^b) - \log(\text{WCSS}_k)
  $$
  Where $( B )$ is the number of bootstrapped random datasets, $( \text{WCSS}^b )$ is the WCSS of the random dataset $( b )$, and $( \text{WCSS}_k )$ is the WCSS for the real data with $( k )$ clusters.

- Choose the smallest $( k )$ such that:
  $$
  \text{Gap}(k) \geq \text{Gap}(k+1) - s_{k+1}
  $$
  Where $( s_{k+1} )$ is the standard deviation of the bootstrapped WCSS estimates.

#### Mathematical Interpretation:
The gap statistic measures how much better the K-Means clustering performs compared to random clustering. The optimal $( k )$ is where the gap statistic is maximized.

### 4. **Davies-Bouldin Index**

The **Davies-Bouldin Index** (DBI) measures the average similarity ratio of each cluster with its most similar cluster. A lower DBI indicates better clustering.

The formula for the DBI for cluster $( i )$ is:
$$
\text{DBI} = \frac{1}{k} \sum_{i=1}^{k} \max_{j \neq i} \left( \frac{\sigma_i + \sigma_j}{d(\mu_i, \mu_j)} \right)
$$
Where:
- $( \sigma_i )$ is the average distance between points in cluster $( i )$ and the centroid $( \mu_i )$,
- $( \mu_i )$ and $( \mu_j )$ are the centroids of clusters $( i )$ and $( j )$,
- $( d(\mu_i, \mu_j) )$ is the distance between the centroids of clusters $( i )$ and $( j )$.

- A lower DBI indicates that clusters are compact and well-separated. You can compute the DBI for various values of $( k )$ and choose the one with the lowest index.

### 5. **BIC (Bayesian Information Criterion) or AIC (Akaike Information Criterion)**

These are model selection criteria typically used in mixture models, but they can also be applied to K-Means to estimate the optimal $( k )$.

#### Bayesian Information Criterion (BIC):
BIC penalizes the likelihood function based on the number of parameters in the model. It is defined as:
$$
\text{BIC}(k) = \log(n) \cdot k - 2 \cdot \log(L)
$$
Where:
- $( n )$ is the number of data points,
- $( k )$ is the number of clusters,
- $( L )$ is the likelihood of the data given the clustering.

#### Akaike Information Criterion (AIC):
AIC is another information criterion that adds a penalty for the number of clusters $( k )$:
$$
\text{AIC}(k) = 2 \cdot k - 2 \cdot \log(L)
$$

For both BIC and AIC, lower values indicate a better balance between goodness of fit and model complexity.

### Conclusion:
- **Elbow Method** is a simple and intuitive method, especially when plotting WCSS vs. $( k )$.
- **Silhouette Score** provides a measure of how well-separated clusters are.
- **Gap Statistic** compares clustering performance with random data.
- **Davies-Bouldin Index** minimizes inter-cluster similarity, indicating well-separated clusters.
- **BIC/AIC** help in balancing model fit and complexity.

Each method has its strengths and weaknesses, and in practice, it's often useful to try multiple methods to validate the choice of $( k )$.