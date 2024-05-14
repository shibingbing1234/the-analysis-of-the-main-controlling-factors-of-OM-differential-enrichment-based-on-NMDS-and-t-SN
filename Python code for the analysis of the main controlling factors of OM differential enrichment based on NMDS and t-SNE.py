# -*- coding: utf-8 -*-
"""
Created on Dec 5 2023

@author: Bingbing Shi
"""

import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.manifold import MDS, TSNE
from scipy.spatial.distance import pdist, cdist, squareform
from scipy.interpolate import griddata
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from sklearn.decomposition import PCA
from sklearn.metrics import silhouette_score, calinski_harabasz_score, davies_bouldin_score
from skopt import gp_minimize
from skopt.space import Integer
import warnings

# Ignore all warnings.
warnings.filterwarnings("ignore")

# Data input
data1 = pd.read_excel('S1 Paleoredox dataset for input.xlsx', index_col=0)
data = data1.iloc[:, :-2]  # Exclude the final two columns (TOC, PG).
parameters = list(data.columns)
regions = data1.index

# 2. Define a function to replace outliers that exceed the upper and lower limits with sub-outliers
def replace_outliers(series):
    Q1 = series.quantile(0.15)
    Q3 = series.quantile(0.85)
    IQR = Q3 - Q1
    lower_boundary = Q1 - 2.0 * IQR
    upper_boundary = Q3 + 2.0 * IQR
    outliers_mask = ~((series >= lower_boundary) & (series <= upper_boundary))
    
    # For each outlier, find the nearest non-outliers and replace with their mean
    for idx in series[outliers_mask].index:
        # Find previous non-outlier
        prev_idx = int(idx) - 1 if idx.isdigit() and int(idx) > 0 else None
        
        # Find next non-outlier
        next_idx = int(idx) + 1 if idx.isdigit() and int(idx) < len(series) else None
        
        # Calculate mean of the nearest non-outliers
        if prev_idx is not None and next_idx is not None:
            series[idx] = (series[prev_idx] + series[next_idx]) / 2
        elif prev_idx is not None:
            series[idx] = series[prev_idx]
        elif next_idx is not None:
            series[idx] = series[next_idx]
    
    return series

# Apply outlier treatment to each parameter. 
for param in parameters:
    data[param] = replace_outliers(data[param])

# Standardize using Z-score. 
scaler = StandardScaler()
normalized_data = pd.DataFrame(scaler.fit_transform(data), columns=data.columns, index=data.index)

def calculate_r2(observed, predicted):
    total_sum_squares = np.sum((observed - np.mean(observed)) ** 2)
    residual_sum_squares = np.sum((observed - predicted) ** 2)
    return 1 - (residual_sum_squares / total_sum_squares)

def evaluate_parameters(parameters, data):
    subset_data = data[parameters]
    dist_matrix = squareform(pdist(subset_data.values))
    nmds = MDS(n_components=2, dissimilarity='precomputed', max_iter=5000, eps=1e-3, n_init=100, metric=False, random_state=42, n_jobs=-1)
    nmds_result = nmds.fit_transform(dist_matrix)
    stress = nmds.stress_
    ordination_dist_matrix = squareform(pdist(nmds_result))
    ordination_distances = ordination_dist_matrix[0]
    observed_dissimilarity = dist_matrix[0]
    non_metric_r2 = calculate_r2(observed_dissimilarity, ordination_distances)
    linear_fit_r2 = calculate_r2(np.polyval(np.polyfit(observed_dissimilarity, ordination_distances, 1), observed_dissimilarity), ordination_distances)
    return stress, nmds_result, observed_dissimilarity, non_metric_r2, linear_fit_r2

def evaluate(parameters):
    stress, _, observed_dissimilarity, non_metric_r2, linear_fit_r2 = evaluate_parameters(parameters, normalized_data)
    fitness = -0.1 * stress - 0.2 * np.sum(observed_dissimilarity) + 0.3 * non_metric_r2 + 0.4 * linear_fit_r2
    return fitness

# Implement Bayesian optimization function.
def optimize_bayes():
    def objective(params):
        print("Params:", params)
        subset = [parameters[i] for i in range(len(params)) if params[i]]
        print("Subset:", subset)
        return -evaluate(subset)

    space = [Integer(0, 1) for _ in range(len(parameters))]
    print("Space dimensions:", len(space))
    res = gp_minimize(objective, space, n_calls=60, random_state=42, verbose=True)
    best_params = [parameters[i] for i in range(len(res.x)) if res.x[i]]
    return res, best_params 

# Optimize the process. 
res, best_parameters = optimize_bayes()

print("Best parameters:", best_parameters)
stress, best_nmds_result, best_observed_dissimilarity, best_non_metric_r2, best_linear_fit_r2 = evaluate_parameters(best_parameters, normalized_data)

print("Best stress:", stress)
print("Non-metric R2:", best_non_metric_r2)
print("Linear fit R2:", best_linear_fit_r2)

#  Plot scatter diagrams.
plt.scatter(best_observed_dissimilarity.flatten(), best_nmds_result[:, 0], c='b', label='Observations')
plt.text(0.5, 0.9, f"Non-metric R2: {best_non_metric_r2:.2f}", transform=plt.gca().transAxes)
plt.text(0.5, 0.85, f"Linear fit R2: {best_linear_fit_r2:.2f}", transform=plt.gca().transAxes)
plt.xlabel('Ordination Distances')
plt.ylabel('Observed Dissimilarity')
plt.title('Correlation between Observed Dissimilarity and Ordination Distances')
plt.legend()
plt.show()

# Add noise to the data.
def add_noise(data, regions, intra_region_noise=0.01, inter_region_noise=0.02):
    noisy_data = data.copy()
    for region in np.unique(regions):
        region_data = noisy_data[regions == region]
        noise = np.random.normal(scale=intra_region_noise, size=region_data.shape)
        noisy_data[regions == region] += noise
    
    global_noise = np.random.normal(scale=inter_region_noise, size=data.shape)
    noisy_data += global_noise
    return noisy_data

from sklearn.cluster import KMeans

# Calculate three scoring metrics and a composite score.
def calculate_scores(labels, data_scaled):
    silhouette = silhouette_score(data_scaled, labels)
    calinski_harabasz = calinski_harabasz_score(data_scaled, labels)
    davies_bouldin = davies_bouldin_score(data_scaled, labels)
    composite = (0.4 * silhouette) + (0.4 * calinski_harabasz) - (0.2 * davies_bouldin)
    return silhouette, calinski_harabasz, davies_bouldin, composite

# Optimize t-SNE parameters. 
def optimize_tsne(data, regions, max_iterations=200, convergence_threshold=1e-4):
    best_score = float('-inf')
    best_tsne_params = None
    best_tsne_result = None
    best_scores = None
    
    scaler = StandardScaler()
    data_scaled = scaler.fit_transform(data)
    
    for perp in np.linspace(20, 40, 12, dtype=int):
        for lr in np.linspace(500, 1500, 23, dtype=int):
            for rs in np.linspace(0, 100, 11, dtype=int):
                tsne = TSNE(n_components=2, perplexity=perp, learning_rate=lr, random_state=rs)
                tsne_result = tsne.fit_transform(data_scaled)
                
                # Perform clustering on the t-SNE result
                kmeans = KMeans(n_clusters=4, random_state=rs)
                tsne_labels = kmeans.fit_predict(tsne_result)
                
                # Compute the scoring metrics.
                silhouette, calinski_harabasz, davies_bouldin, composite = calculate_scores(tsne_labels, data_scaled)
                
                # Update the best scores and parameters.
                if composite > best_score:
                    best_score = composite
                    best_tsne_params = (perp, lr, rs)
                    best_tsne_result = tsne_result
                    best_scores = silhouette, calinski_harabasz, davies_bouldin

    return best_tsne_params, best_tsne_result, best_scores, best_score

scaler = StandardScaler()
data_scaled = scaler.fit_transform(data)

# Perform PCA dimensionality reduction.
pca = PCA(n_components=2)
pca_result = pca.fit_transform(data_scaled)

# Execute k-means clustering. 
kmeans = KMeans(n_clusters=4)
cluster_labels = kmeans.fit_predict(pca_result)

# Compute the scoring metrics.
pca_silhouette = silhouette_score(data_scaled, cluster_labels)
pca_calinski_harabasz = calinski_harabasz_score(data_scaled, cluster_labels)
pca_davies_bouldin = davies_bouldin_score(data_scaled, cluster_labels)

# Print the scoring metrics.
print(f"PCA Silhouette Score: {pca_silhouette}")
print(f"PCA Calinski-Harabasz Score: {pca_calinski_harabasz}")
print(f"PCA Davies-Bouldin Score: {pca_davies_bouldin}")


# Compute distance value
def compute_distance_value(data, regions, metric='euclidean'):
    unique_regions = np.unique(regions)
    n_clusters = len(unique_regions)

    intra_cluster_distances = np.mean([np.mean(pdist(data[regions == region], metric=metric)) for region in unique_regions])

    inter_cluster_distances = 0
    for i in range(n_clusters):
        for j in range(i + 1, n_clusters):
            inter_cluster_distances += np.mean(cdist(data[regions == unique_regions[i]], data[regions == unique_regions[j]], metric=metric))
    inter_cluster_distances /= (n_clusters * (n_clusters - 1) / 2)

    return inter_cluster_distances - intra_cluster_distances

# Plot t-SNE scatter plot with best parameters
valid_params = list(best_parameters)
valid_data = normalized_data[valid_params]
best_tsne_params, tsne_result, best_scores, best_score = optimize_tsne(valid_data, regions)
# 打印最佳评分指标
tsne_silhouette, tsne_calinski_harabasz, tsne_davies_bouldin = best_scores
print(f"t-SNE Silhouette Score: {tsne_silhouette}")
print(f"t-SNE Calinski-Harabasz Score: {tsne_calinski_harabasz}")
print(f"t-SNE Davies-Bouldin Score: {tsne_davies_bouldin}")
print(f"Composite Score: {best_score}")
region_mapping = {
    'DHS': ('red', 'o'),
    'GDK': ('blue', 'x'),
    'HYC': ('green', 's'),
    'JJZG': ('black', '^'),
}

legend_elements = [patches.Patch(facecolor=value[0], edgecolor='black', label=key) for key, value in region_mapping.items()]

plt.subplot(1, 2, 2)
for region, coords in zip(regions, tsne_result):
    color, marker = region_mapping.get(region, ('gray', 'o'))
    plt.scatter(coords[0], coords[1], color=color, marker=marker)
plt.title(f"t-SNE for Regions (perplexity={best_tsne_params[0]}, learning_rate={best_tsne_params[1]})")
plt.legend(handles=legend_elements, loc="upper right")
plt.tight_layout()
plt.show()

# Creating Contour Filled Color Maps
selected_params = valid_params[:2]

# Defining a Grid for Interpolation
grid_x, grid_y = np.mgrid[min(tsne_result[:, 0]):max(tsne_result[:, 0]):100j, 
                          min(tsne_result[:, 1]):max(tsne_result[:, 1]):100j]

for selected_param in selected_params:
    values = normalized_data[selected_param].values
    
    grid_values = griddata(tsne_result, values, (grid_x, grid_y), method='cubic')

    plt.figure(figsize=(8, 6))
    plt.contourf(grid_x, grid_y, grid_values, levels=15, cmap='rainbow')
    plt.colorbar(label=selected_param)
    plt.scatter(tsne_result[:, 0], tsne_result[:, 1], c='black', s=10, marker='o', alpha=0.5)
    plt.title(f"Contour plot for {selected_param} based on t-SNE")
    plt.show()

# Accessing Data
data2 = data1.iloc[:, -2:]  
parameters2 = list(data2.columns)
regions = data1.index
grid_x, grid_y = np.mgrid[min(tsne_result[:, 0]):max(tsne_result[:, 0]):100j,
                        min(tsne_result[:, 1]):max(tsne_result[:, 1]):100j]

for selected_param2 in parameters2:
    values = data2[selected_param2].values  #(Retrieve TOC and PG columns.)
    grid_values = griddata(tsne_result, values, (grid_x, grid_y), method='cubic')
    
    plt.figure(figsize=(8, 6))
    plt.contourf(grid_x, grid_y, grid_values, levels=15, cmap='rainbow')
    plt.colorbar(label=selected_param2)
    plt.scatter(tsne_result[:, 0], tsne_result[:, 1], c='black', s=10, marker='o', alpha=0.5)
    plt.title(f"Contour plot for {selected_param2} based on t-SNE")
    plt.show()

