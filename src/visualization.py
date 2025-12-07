"""
visualization.py
Fonctions pour visualiser les données et résultats
"""

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix

def setup_plotting():
    """Configure les paramètres de plot"""
    plt.style.use('seaborn-v0_8')
    sns.set_palette("husl")
    plt.rcParams['figure.figsize'] = (10, 6)
    plt.rcParams['font.size'] = 12

def plot_digits_samples(digits, n_samples=10):
    """
    Affiche des exemples d'images du dataset

    Parameters:
    -----------
    digits : dataset sklearn
        Dataset des chiffres
    n_samples : int
        Nombre d'échantillons à afficher
    """
    fig, axes = plt.subplots(2, 5, figsize=(12, 5))
    for i, ax in enumerate(axes.flat):
        if i < n_samples:
            ax.imshow(digits.images[i], cmap='gray')
            ax.set_title(f"Vrai: {digits.target[i]}")
        ax.axis('off')

    plt.suptitle(f"Exemples d'images de chiffres (8×8 pixels)", fontsize=14)
    plt.tight_layout()
    return fig

def plot_pca_viz(X_pca, labels, title="Visualisation PCA"):
    """
    Visualise les données dans l'espace PCA 2D

    Parameters:
    -----------
    X_pca : array
        Données réduites en 2D par PCA
    labels : array
        Labels pour la colorisation
    title : str
        Titre du graphique
    """
    fig, ax = plt.subplots(figsize=(10, 8))
    scatter = ax.scatter(X_pca[:, 0], X_pca[:, 1],
                         c=labels, cmap='tab10',
                         s=30, alpha=0.7, edgecolor='k', linewidth=0.5)

    plt.colorbar(scatter, ticks=range(10))
    ax.set_xlabel('Première Composante Principale')
    ax.set_ylabel('Deuxième Composante Principale')
    ax.set_title(title)
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    return fig

def plot_clusters_comparison(X_pca, clusters_pred, clusters_true):
    """
    Compare clusters prédits vs vrais classes

    Parameters:
    -----------
    X_pca : array
        Données réduites en 2D
    clusters_pred : array
        Clusters prédits par K-Means
    clusters_true : array
        Vraies classes
    """
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))

    # Clusters prédits
    scatter1 = axes[0].scatter(X_pca[:, 0], X_pca[:, 1],
                               c=clusters_pred, cmap='tab10',
                               s=30, alpha=0.7, edgecolor='k', linewidth=0.5)
    plt.colorbar(scatter1, ax=axes[0], ticks=range(10))
    axes[0].set_title('Clusters prédits par K-Means')
    axes[0].set_xlabel('PC1')
    axes[0].set_ylabel('PC2')
    axes[0].grid(True, alpha=0.3)

    # Vraies classes
    scatter2 = axes[1].scatter(X_pca[:, 0], X_pca[:, 1],
                               c=clusters_true, cmap='tab10',
                               s=30, alpha=0.7, edgecolor='k', linewidth=0.5)
    plt.colorbar(scatter2, ax=axes[1], ticks=range(10))
    axes[1].set_title('Vraies classes')
    axes[1].set_xlabel('PC1')
    axes[1].set_ylabel('PC2')
    axes[1].grid(True, alpha=0.3)

    plt.suptitle('Comparaison: Clusters prédits vs Vraies classes', fontsize=14)
    plt.tight_layout()
    return fig

def plot_centroids(centroids, mapping=None):
    """
    Visualise les centroïdes des clusters

    Parameters:
    -----------
    centroids : array
        Centroïdes des clusters (shape: (10, 64))
    mapping : dict
        Correspondance cluster → chiffre
    """
    fig, axes = plt.subplots(2, 5, figsize=(15, 6))

    for i, ax in enumerate(axes.flat):
        centroid_img = centroids[i].reshape(8, 8)
        ax.imshow(centroid_img, cmap='gray_r', interpolation='nearest')

        title = f'Cluster {i}'
        if mapping and i in mapping:
            title += f'\n(→ {mapping[i]})'
        ax.set_title(title, fontsize=12)
        ax.axis('off')

    plt.suptitle('Centroïdes des clusters (chiffre moyen de chaque groupe)',
                 fontsize=16, y=1.02)
    plt.tight_layout()
    return fig

def plot_confusion_matrix(y_true, y_pred, accuracy):
    """
    Affiche la matrice de confusion

    Parameters:
    -----------
    y_true : array
        Vrais labels
    y_pred : array
        Labels prédits
    accuracy : float
        Précision pour le titre
    """
    cm = confusion_matrix(y_true, y_pred)

    fig, ax = plt.subplots(figsize=(10, 8))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=ax)

    ax.set_xlabel('Chiffre prédit')
    ax.set_ylabel('Chiffre réel')
    ax.set_title(f'Matrice de confusion (Précision: {accuracy:.2%})')

    plt.tight_layout()
    return fig

def plot_metrics_comparison(metrics_without_pca, metrics_with_pca):
    """
    Compare les métriques avec/sans PCA

    Parameters:
    -----------
    metrics_without_pca : dict
        Métriques sans PCA
    metrics_with_pca : dict
        Métriques avec PCA
    """
    methods = ['Sans PCA', 'Avec PCA']
    accuracies = [metrics_without_pca['accuracy'], metrics_with_pca['accuracy']]

    fig, ax = plt.subplots(figsize=(8, 6))
    bars = ax.bar(methods, accuracies, color=['skyblue', 'lightgreen'])

    ax.set_title('Comparaison des performances')
    ax.set_ylabel('Précision')
    ax.set_ylim(0, 1)
    ax.grid(True, alpha=0.3, axis='y')

    # Ajouter les valeurs
    for bar, acc in zip(bars, accuracies):
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2, height + 0.02,
               f'{acc:.2%}', ha='center', va='bottom', fontweight='bold')

    plt.tight_layout()
    return fig