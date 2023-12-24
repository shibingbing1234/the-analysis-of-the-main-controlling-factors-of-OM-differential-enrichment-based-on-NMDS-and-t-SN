# -*- coding: utf-8 -*-
"""
Created on Tue Aug  1 16:15:04 2023

@author: Y
"""
import pandas as pd
import numpy as np
from itertools import combinations
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.manifold import MDS, TSNE
from sklearn.metrics import silhouette_score
from scipy.spatial.distance import pdist, cdist, squareform
from scipy.interpolate import griddata
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from sklearn.metrics import r2_score
from scipy.stats import linregress
from tqdm import tqdm
from sklearn.decomposition import PCA
from deap import base, creator, tools
import random
normalized_stress = 'auto'
import warnings
warnings.filterwarnings("ignore", category=FutureWarning, module="sklearn.manifold._mds")
# Loading Excel Data
data3 = pd.read_excel('Paleosalinity.xlsx', index_col=0)
data = data3.iloc[:, :-2]  # Exclude the final two columns
parameters = list(data.columns)  # Exclude the final two columns
regions = data3.index   
def replace_outliers(series):
    lower_percentile = series.quantile(0.05)
    upper_percentile = series.quantile(0.95)
    series = np.where(series < lower_percentile, lower_percentile, series)
    series = np.where(series > upper_percentile, upper_percentile, series)
    return series

for param in parameters:
    data[param] = replace_outliers(data[param])

normalized_data = (data - data.min()) / (data.max() - data.min())

def calculate_r2(x, y):
    total_sum_squares = np.sum((y - np.mean(y)) ** 2)
    residual_sum_squares = np.sum((y - x) ** 2)
    r2 = 1 - (residual_sum_squares / total_sum_squares)
    return r2

def evaluate_parameters(parameters_subset, data):
    subset_data = data.loc[:, list(parameters_subset)]
    
    # Determine the distance between each row and the first row
    dist_matrix = squareform(pdist(subset_data.values))
    observed_dissimilarity = dist_matrix[0]
    
    nmds = MDS(
        n_components=2,
        dissimilarity='precomputed',
        max_iter=10000,
        eps=1e-30,
        n_init=5000,
        metric=False,
        random_state=42
    )
    
    nmds_result = nmds.fit_transform(dist_matrix)
    stress = nmds.stress_
    
    # Compute the relative distance between each sample point and the first point
    ordination_dist_matrix = squareform(pdist(nmds_result))
    ordination_distances = ordination_dist_matrix[0]
    
    non_metric_r2 = calculate_r2(observed_dissimilarity, ordination_distances)
    linear_fit_r2 = calculate_r2(
        np.polyval(np.polyfit(observed_dissimilarity, ordination_distances, 1), observed_dissimilarity),
        ordination_distances
    )
    
    return stress, nmds_result, observed_dissimilarity, non_metric_r2, linear_fit_r2

# Find best parameters
# Define the fitness function
# Define the fitness function
def evaluate(subset):
    stress, mds_result, observed_dissimilarity, non_metric_r2, linear_fit_r2 = evaluate_parameters(subset, normalized_data)
    # Assume that stress and observed_dissimilarity should be minimized, and the others should be maximized
    fitness = -0.1*stress - 0.2*observed_dissimilarity + 0.3*non_metric_r2 + 0.4*linear_fit_r2
    return fitness,

# Create the DEAP toolbox
toolbox = base.Toolbox()

# Define the individual and population
creator.create("FitnessMax", base.Fitness, weights=(1.0,))  # Set four weights
creator.create("Individual", list, fitness=creator.FitnessMax)

# Register the attr_bool function (which generates either True or False)
toolbox.register("attr_bool", random.choice, [True, False])

# Register the individual and population functions
toolbox.register("individual", tools.initRepeat, creator.Individual, toolbox.attr_bool, n=len(parameters))
toolbox.register("population", tools.initRepeat, list, toolbox.individual)

# Register the genetic operators
toolbox.register("evaluate", evaluate)
toolbox.register("mate", tools.cxTwoPoint)
toolbox.register("mutate", tools.mutFlipBit, indpb=0.05)
toolbox.register("select", tools.selTournament, tournsize=3)

# Set some variables to maintain the best solution
best_individual = None
best_fitness = -1
best_non_metric_r2 = -1
best_linear_fit_r2 = -1
best_observed_dissimilarity = None
best_nmds_result = None
best_parameters = None
# Initialize a population
pop = toolbox.population(n=50)  # 50 individuals

# Begin the evolution
with tqdm(total=100) as pbar:  # 100 generations
    for gen in range(100):
        offspring = toolbox.select(pop, len(pop))
        offspring = list(map(toolbox.clone, offspring))

        # Apply crossover and mutation
        for child1, child2 in zip(offspring[::2], offspring[1::2]):
            if random.random() < 0.5:  # 50% chance to cross two individuals
                toolbox.mate(child1, child2)
                del child1.fitness.values
                del child2.fitness.values

        for mutant in offspring:
            if random.random() < 0.2:  # 20% chance to mutate an individual
                toolbox.mutate(mutant)
                del mutant.fitness.values

        # Evaluate the individuals with invalid fitness
        invalid_ind = [ind for ind in offspring if not ind.fitness.valid]
        fitnesses = map(toolbox.evaluate, invalid_ind)

        # Update the best solution
        for ind, fit in zip(invalid_ind, fitnesses):
            ind.fitness.values = fit

            if fit[0][0] > best_fitness:
                best_fitness = fit[0]
                best_individual = ind
                # Re-evaluate the best individual to get all the metrics
                stress, mds_result, observed_dissimilarity, non_metric_r2, linear_fit_r2 = evaluate_parameters(best_individual, normalized_data)
                best_non_metric_r2 = non_metric_r2
                best_linear_fit_r2 = linear_fit_r2
                best_observed_dissimilarity = observed_dissimilarity
                best_nmds_result = mds_result
                best_parameters =best_individual
            pop[:] = offspring

        pbar.update()

# Print the best solution
print("Best fitness:", best_fitness)
print("Best parameter combination:",best_parameters)
print("Non-metric R2:", best_non_metric_r2)
print("Linear fit R2:", best_linear_fit_r2)

# Plot scatter plot
plt.scatter(best_observed_dissimilarity.flatten(), best_nmds_result[:, 0], c='b', label='Observations')
plt.text(0.5, 0.9, f"Non-metric R2: {best_non_metric_r2:.2f}", transform=plt.gca().transAxes)
plt.text(0.5, 0.85, f"Linear fit R2: {best_linear_fit_r2:.2f}", transform=plt.gca().transAxes)
plt.xlabel('Ordination Distances')
plt.ylabel('Observed Dissimilarity')
plt.title('Correlation between Observed Dissimilarity and Ordination Distances')
plt.legend()
plt.show()
# Add noise to the data
def add_noise(data, regions, intra_region_noise=0.01, inter_region_noise=0.02):

    noisy_data = data.copy()
    
    # Add intra-region noise
    for region in regions.unique():
        region_data = noisy_data[regions == region]
        noise = np.random.normal(scale=intra_region_noise, size=region_data.shape)
        noisy_data[regions == region] += noise
    
    # Add inter-region noise
    global_noise = np.random.normal(scale=inter_region_noise, size=data.shape)
    noisy_data += global_noise
    
    return noisy_data

# Optimize t-SNE
# Optimize t-SNE
def optimize_tsne(data, regions):
    best_score = float('-inf')  # We want to maximize inter_cluster_distances - intra_cluster_distances
    best_tsne_params = None
    best_tsne_result = None

    scaler = StandardScaler()
    data_scaled = scaler.fit_transform(data)

    for n_components in range(2, 4, 1):  # Try different number of components for PCA
        pca = PCA(n_components=n_components)
        data_pca = pca.fit_transform(data_scaled)

        for perp in range(30, 150, 10):  # Try different perplexity parameter values
            for lr in range(500, 1500, 30):  # Try different learning rate parameter values
                for rs in range(0, 101, 10):  # Try different random seeds
                    noisy_data = add_noise(data_pca, regions)
                    noisy_data_scaled = scaler.transform(noisy_data)  # Scale the noisy data
                    tsne = TSNE(n_components=2, perplexity=perp, learning_rate=lr, random_state=rs)
                    tsne_result = tsne.fit_transform(noisy_data_scaled)

                    score = compute_distance_value(tsne_result, regions)  # Use compute_distance_value function to evaluate

                    if score > best_score:  # We want to maximize inter_cluster_distances - intra_cluster_distances
                        best_score = score
                        best_tsne_params = (n_components, perp, lr, rs)
                        best_tsne_result = tsne_result

    return best_tsne_params, best_tsne_result

# Compute distance value
def compute_distance_value(data, regions, metric='euclidean'):
    unique_regions = np.unique(regions)
    n_clusters = len(unique_regions)

    intra_cluster_distances = np.mean([np.mean(pdist(data[regions == region], metric=metric)) for region in unique_regions])

    inter_cluster_distances = 0
    for i in range(n_clusters):
        for j in range(i+1, n_clusters):
            inter_cluster_distances += np.mean(cdist(data[regions == unique_regions[i]], data[regions == unique_regions[j]], metric=metric))
    inter_cluster_distances /= (n_clusters * (n_clusters - 1) / 2)

    return inter_cluster_distances - intra_cluster_distances

# Plot t-SNE scatter plot with best parameters
valid_params = list(best_parameters)
valid_data = normalized_data[valid_params]
best_tsne_params, tsne_result = optimize_tsne(valid_data, regions)

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

#Creating Contour Filled Color Maps
selected_params = valid_params[:2]  # Choose the first two valid parameters for demonstration

# Defining a Grid for Interpolation




grid_x, grid_y = np.mgrid[min(tsne_result[:, 0]):max(tsne_result[:, 0]):100j, 
                          min(tsne_result[:, 1]):max(tsne_result[:, 1]):100j]

for selected_param in selected_params:
    values = normalized_data[selected_param].values
    
    # Performing interpolation
    grid_values = griddata(tsne_result, values, (grid_x, grid_y), method='cubic')

    plt.figure(figsize=(8, 6))
    plt.contourf(grid_x, grid_y, grid_values, levels=15, cmap='rainbow')
    plt.colorbar(label=selected_param)
    plt.scatter(tsne_result[:, 0], tsne_result[:, 1], c='black', s=10, marker='o', alpha=0.5)
    plt.title(f"Contour plot for {selected_param} based on t-SNE")
    plt.show()
   
    

# Accessing Data
data2 = pd.read_excel('Paleoredox.xlsx', index_col=0)
parameters2 = list(data2.columns[-2:])  # Obtain the last two columns
regions = data2.index

# Definition of Parameters to be Plotted
selected_params2 = parameters2[:2]  # Select the first two parameters to demonstrate
# Defined interpolation grid
grid_x, grid_y = np.mgrid[min(tsne_result[:, 0]):max(tsne_result[:, 0]):100j,
                        min(tsne_result[:, 1]):max(tsne_result[:, 1]):100j]

# Draw contour fill color map
for selected_param in selected_params2:
    values = normalized_data[selected_param].values
    grid_values = griddata(tsne_result, values, (grid_x, grid_y), method='cubic')
    
    plt.figure(figsize=(8, 6))
    plt.contourf(grid_x, grid_y, grid_values, levels=15, cmap='rainbow')
    plt.colorbar(label=selected_param)
    plt.scatter(tsne_result[:, 0], tsne_result[:, 1], c='black', s=10, marker='o', alpha=0.5)
    plt.title(f"Contour plot for {selected_param} based on t-SNE")
    plt.show()