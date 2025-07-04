from sklearn.cluster import KMeans
import numpy as np
import pandas as pd

def kmeans(self, batch_size):
    if len(self.pool_df) > batch_size:
        # Ensure 'SMILES' is set as the index for efficient lookup
        fingerprint_df_indexed = self.fingerprint_df.set_index('SMILES')
        # Retrieve fingerprints corresponding to 'pool_df' SMILES
        fingerprints = fingerprint_df_indexed.reindex(self.pool_df['SMILES'])['fingerprint'].values
        # Stack fingerprints into a 2D array
        X = np.vstack(fingerprints)
        # Fit KMeans clustering
        km = KMeans(init='k-means++', n_clusters=batch_size, n_init='auto')
        km.fit(X)
        # Compute distances to assigned centroids
        distances = np.linalg.norm(X - km.cluster_centers_[km.labels_], axis=1)
        # Create a DataFrame with SMILES, cluster labels, and distances
        data = pd.DataFrame({
            'SMILES': self.pool_df['SMILES'].values,
            'cluster': km.labels_,
            'distance': distances
        })
        # Select SMILES closest to each centroid
        selected_smiles = data.loc[data.groupby('cluster')['distance'].idxmin(), 'SMILES'].values
    else:
        # If pool size is less than or equal to batch_size, select all SMILES
        selected_smiles = self.pool_df['SMILES'].values
    return selected_smiles
