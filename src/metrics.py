"""
metrics.py
Fonctions pour l'évaluation des clusters
"""

import numpy as np
from scipy.optimize import linear_sum_assignment
from sklearn.metrics import (precision_score, recall_score,
                           f1_score, classification_report)

def hungarian_mapping(cluster_labels, true_labels):
    """
    Trouve la meilleure correspondance entre clusters et classes
    en utilisant l'algorithme hongrois

    Parameters:
    -----------
    cluster_labels : array
        Labels des clusters (0-9)
    true_labels : array
        Vrais labels (0-9)

    Returns:
    --------
    clusters_mapped : array
        Clusters mappés aux vraies classes
    mapping : dict
        Dictionnaire de correspondance
    """
    n_clusters = len(np.unique(cluster_labels))
    n_classes = len(np.unique(true_labels))

    # Matrice de contingence
    contingency_matrix = np.zeros((n_clusters, n_classes))

    for i in range(len(cluster_labels)):
        contingency_matrix[cluster_labels[i], true_labels[i]] += 1

    # Algorithme hongrois pour maximiser les correspondances
    row_ind, col_ind = linear_sum_assignment(-contingency_matrix)

    # Création du mapping
    mapping = {int(row): int(col) for row, col in zip(row_ind, col_ind)}
    clusters_mapped = np.array([mapping[label] for label in cluster_labels])

    return clusters_mapped, mapping

def evaluate_clusters(y_true, y_pred_mapped):
    """
    Évalue les performances des clusters

    Parameters:
    -----------
    y_true : array
        Vrais labels
    y_pred_mapped : array
        Labels prédits après mapping

    Returns:
    --------
    metrics : dict
        Dictionnaire des métriques
    """
    accuracy = np.mean(y_pred_mapped == y_true)

    # Métriques par classe
    precision_per_class = []
    recall_per_class = []
    f1_per_class = []

    for digit in range(10):
        y_true_binary = (y_true == digit).astype(int)
        y_pred_binary = (y_pred_mapped == digit).astype(int)

        precision = precision_score(y_true_binary, y_pred_binary, zero_division=0)
        recall = recall_score(y_true_binary, y_pred_binary, zero_division=0)
        f1 = f1_score(y_true_binary, y_pred_binary, zero_division=0)

        precision_per_class.append(precision)
        recall_per_class.append(recall)
        f1_per_class.append(f1)

    metrics = {
        'accuracy': accuracy,
        'precision_per_class': precision_per_class,
        'recall_per_class': recall_per_class,
        'f1_per_class': f1_per_class,
        'precision_mean': np.mean(precision_per_class),
        'recall_mean': np.mean(recall_per_class),
        'f1_mean': np.mean(f1_per_class)
    }

    return metrics

def print_cluster_analysis(clusters, y_true, clusters_mapped, mapping, metrics):
    """
    Affiche une analyse détaillée des clusters

    Parameters:
    -----------
    clusters : array
        Clusters prédits
    y_true : array
        Vrais labels
    clusters_mapped : array
        Clusters après mapping
    mapping : dict
        Correspondance cluster → chiffre
    metrics : dict
        Métriques d'évaluation
    """
    print("="*60)
    print(" ANALYSE DES CLUSTERS")
    print("="*60)

    # Correspondance
    print("\n CORRESPONDANCE CLUSTERS → CHIFFRES:")
    for cluster, digit in sorted(mapping.items()):
        print(f"   Cluster {cluster} → Chiffre {digit}")

    # Distribution
    print(f"\n DISTRIBUTION DES CLUSTERS:")
    cluster_counts = np.bincount(clusters)
    for i in range(10):
        print(f"   Cluster {i}: {cluster_counts[i]:4d} images")

    # Précision globale
    print(f"\n PERFORMANCE GLOBALE:")
    print(f"   Précision: {metrics['accuracy']:.2%}")
    print(f"   Précision moyenne: {metrics['precision_mean']:.2%}")
    print(f"   Rappel moyen: {metrics['recall_mean']:.2%}")
    print(f"   F1-score moyen: {metrics['f1_mean']:.2%}")

    # Détail par chiffre
    print(f"\n DÉTAIL PAR CHIFFRE:")
    print("   Chiffre | Précision | Rappel | F1-score | Support")
    print("   " + "-"*45)
    for i in range(10):
        support = np.sum(y_true == i)
        print(f"   {i:7d} | {metrics['precision_per_class'][i]:9.2%} | "
              f"{metrics['recall_per_class'][i]:6.2%} | "
              f"{metrics['f1_per_class'][i]:8.2%} | {support:7d}")

    print("="*60)