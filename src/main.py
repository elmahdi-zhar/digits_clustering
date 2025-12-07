

import numpy as np
from sklearn.datasets import load_digits
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
from sklearn.metrics import accuracy_score
# Import des modules locaux
from src import metrics
from src import visualization
import os


def create_directories():

    directories = [
        'reports/figures',
        'data',
        'notebooks'
    ]

    for directory in directories:
        os.makedirs(directory, exist_ok=True)






def main():
    """
    Fonction principale exécutant tout le pipeline
    """
    print("="*60)
    print(" PROJET: CLUSTERING DES CHIFFRES MANUSCRITS")
    print("="*60)
    create_directories()
    # Initialisation
    visualization.setup_plotting()

    # ========== ÉTAPE 1: CHARGEMENT DES DONNÉES ==========
    print("\n ÉTAPE 1: CHARGEMENT DES DONNÉES")
    print("-"*40)

    digits = load_digits()
    X = digits.data
    y = digits.target

    print(f"✅ Dataset chargé")
    print(f"   • Nombre d'images: {X.shape[0]}")
    print(f"   • Dimensions: {X.shape[1]} pixels (8×8)")
    print(f"   • Classes: {len(np.unique(y))} chiffres (0-9)")

    # Visualiser des exemples
    fig1 = visualization.plot_digits_samples(digits)
    fig1.savefig('reports/figures/digits_samples.png', dpi=150, bbox_inches='tight')

    # ========== ÉTAPE 2: PRÉPARATION DES DONNÉES ==========
    print("\n ÉTAPE 2: PRÉPARATION DES DONNÉES")
    print("-"*40)

    # Normalisation
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    print(f"✅ Données normalisées")
    print(f"   • Avant: moyenne={X.mean():.1f}, écart-type={X.std():.1f}")
    print(f"   • Après: moyenne={X_scaled.mean():.3f}, écart-type={X_scaled.std():.3f}")

    # PCA pour visualisation 2D
    pca_2d = PCA(n_components=2, random_state=42)
    X_pca_2d = pca_2d.fit_transform(X_scaled)

    fig2 = visualization.plot_pca_viz(X_pca_2d, y, "Vraies classes des chiffres")
    fig2.savefig('reports/figures/pca_true_classes.png', dpi=150, bbox_inches='tight')

    # ========== ÉTAPE 3: K-MEANS SANS PCA ==========
    print("\n ÉTAPE 3: CLUSTERING K-MEANS (sans PCA)")
    print("-"*40)

    kmeans_basic = KMeans(n_clusters=10, random_state=42, n_init=20)
    kmeans_basic.fit(X_scaled)
    clusters_basic = kmeans_basic.predict(X_scaled)

    print(f"✅ K-Means terminé")
    print(f"   • Inertie: {kmeans_basic.inertia_:.2f}")
    print(f"   • Itérations: {kmeans_basic.n_iter_}")

    # Mapping et évaluation
    clusters_mapped_basic, mapping_basic = metrics.hungarian_mapping(clusters_basic, y)
    metrics_basic = metrics.evaluate_clusters(y, clusters_mapped_basic)
    # ==== CALCUL PRÉCISION BRUTE + GAIN (sans PCA) ====

    accuracy_raw_basic = accuracy_score(y, clusters_basic)  # précision brute
    accuracy_hungarian_basic = metrics_basic['accuracy']  # précision après Hungarian
    gain_basic = (accuracy_hungarian_basic - accuracy_raw_basic) * 100  # gain %

    print("\n  COMPARAISON AVANT / APRÈS REMAPPING (Sans PCA)")
    print("  --------------------------------------------------")
    print(f"   Précision brute :         {accuracy_raw_basic:.2%}")
    print(f"   Après Hungarian :         {accuracy_hungarian_basic:.2%}")
    print(f"   Gain :                    {gain_basic:.2f}%")
    print("  --------------------------------------------------")

    # Visualisation
    fig3 = visualization.plot_clusters_comparison(X_pca_2d, clusters_basic, y)
    fig3.savefig('reports/figures/clusters_basic_comparison.png', dpi=150, bbox_inches='tight')

    # Centroïdes
    centroids_basic = scaler.inverse_transform(kmeans_basic.cluster_centers_)
    fig4 = visualization.plot_centroids(centroids_basic, mapping_basic)
    fig4.savefig('reports/figures/centroids_basic.png', dpi=150, bbox_inches='tight')

    # Matrice de confusion
    fig5 = visualization.plot_confusion_matrix(y, clusters_mapped_basic, metrics_basic['accuracy'])
    fig5.savefig('reports/figures/confusion_basic.png', dpi=150, bbox_inches='tight')

    # ========== ÉTAPE 4: K-MEANS AVEC PCA ==========
    print("\n ÉTAPE 4: CLUSTERING K-MEANS (avec PCA)")
    print("-"*40)

    # Réduction de dimension avec PCA
    pca_reduced = PCA(n_components=0.95, random_state=42)
    X_pca_reduced = pca_reduced.fit_transform(X_scaled)

    print(f"✅ PCA appliquée")
    print(f"   • Dimensions: {X.shape[1]} → {X_pca_reduced.shape[1]}")
    print(f"   • Variance conservée: {pca_reduced.explained_variance_ratio_.sum():.2%}")

    # K-Means sur données réduites
    kmeans_pca = KMeans(n_clusters=10, random_state=42, n_init=20)
    kmeans_pca.fit(X_pca_reduced)
    clusters_pca = kmeans_pca.predict(X_pca_reduced)

    print(f"✅ K-Means sur données réduites")
    print(f"   • Inertie: {kmeans_pca.inertia_:.2f}")
    print(f"   • Itérations: {kmeans_pca.n_iter_}")

    # Mapping et évaluation
    clusters_mapped_pca, mapping_pca = metrics.hungarian_mapping(clusters_pca, y)
    metrics_pca = metrics.evaluate_clusters(y, clusters_mapped_pca)
    # ==== CALCUL PRÉCISION BRUTE + GAIN (avec PCA) ====

    accuracy_raw_pca = accuracy_score(y, clusters_pca)  # précision brute
    accuracy_hungarian_pca = metrics_pca['accuracy']  # après Hungarian
    gain_pca = (accuracy_hungarian_pca - accuracy_raw_pca) * 100  # gain %

    print("\n  COMPARAISON AVANT / APRÈS REMAPPING (Avec PCA)")
    print("  --------------------------------------------------")
    print(f"   Précision brute :         {accuracy_raw_pca:.2%}")
    print(f"   Après Hungarian :         {accuracy_hungarian_pca:.2%}")
    print(f"   Gain :                    {gain_pca:.2f}%")
    print("  --------------------------------------------------")

    # Centroïdes (attention: besoin de retransformer)
    centroids_pca_space = kmeans_pca.cluster_centers_
    centroids_pca = pca_reduced.inverse_transform(centroids_pca_space)
    centroids_pca = scaler.inverse_transform(centroids_pca)

    fig6 = visualization.plot_centroids(centroids_pca, mapping_pca)
    fig6.savefig('reports/figures/centroids_pca.png', dpi=150, bbox_inches='tight')

    # Matrice de confusion
    fig7 = visualization.plot_confusion_matrix(y, clusters_mapped_pca, metrics_pca['accuracy'])
    fig7.savefig('reports/figures/confusion_pca.png', dpi=150, bbox_inches='tight')

    # ========== ÉTAPE 5: COMPARAISON ET CONCLUSION ==========
    print("\n ÉTAPE 5: COMPARAISON DES RÉSULTATS")
    print("-"*40)

    # Affichage des analyses
    print("\n ANALYSE SANS PCA:")
    metrics.print_cluster_analysis(clusters_basic, y, clusters_mapped_basic,
                                   mapping_basic, metrics_basic)

    print("\n ANALYSE AVEC PCA:")
    metrics.print_cluster_analysis(clusters_pca, y, clusters_mapped_pca,
                                   mapping_pca, metrics_pca)

    # Comparaison graphique
    fig8 = visualization.plot_metrics_comparison(metrics_basic, metrics_pca)
    fig8.savefig('reports/figures/comparison_metrics.png', dpi=150, bbox_inches='tight')

    # ========== SAUVEGARDE DES RÉSULTATS ==========
    print("\n SAUVEGARDE DES RÉSULTATS")
    print("-"*40)

    with open('reports/results.txt', 'w', encoding='utf-8') as f:
        f.write("="*60 + "\n")
        f.write("RÉSULTATS DU CLUSTERING DES CHIFFRES\n")
        f.write("="*60 + "\n\n")

        f.write("SANS PCA:\n")
        f.write(f"  Précision brute: {accuracy_raw_basic:.2%}\n")
        f.write(f"  Gain après Hungarian: {gain_basic:.2f}%\n\n")

        f.write(f"  Précision: {metrics_basic['accuracy']:.2%}\n")
        f.write(f"  Précision moyenne: {metrics_basic['precision_mean']:.2%}\n")
        f.write(f"  Rappel moyen: {metrics_basic['recall_mean']:.2%}\n")
        f.write(f"  F1-score moyen: {metrics_basic['f1_mean']:.2%}\n\n")

        f.write("AVEC PCA (95% variance):\n")
        f.write(f"  Précision brute: {accuracy_raw_pca:.2%}\n")
        f.write(f"  Gain après Hungarian: {gain_pca:.2f}%\n\n")

        f.write(f"  Précision: {metrics_pca['accuracy']:.2%}\n")
        f.write(f"  Précision moyenne: {metrics_pca['precision_mean']:.2%}\n")
        f.write(f"  Rappel moyen: {metrics_pca['recall_mean']:.2%}\n")
        f.write(f"  F1-score moyen: {metrics_pca['f1_mean']:.2%}\n\n")

        f.write("AMÉLIORATION:\n")
        improvement = metrics_pca['accuracy'] - metrics_basic['accuracy']
        f.write(f"  Différence: {improvement:+.2%}\n")

        if improvement > 0:
            f.write("  ✅ PCA améliore les performances\n")
        else:
            f.write("  ⚠️  PCA ne améliore pas les performances\n")

    print(f"✅ Figures sauvegardées dans: reports/figures/")
    print(f"✅ Résultats sauvegardés dans: reports/results.txt")

    print("\n" + "="*60)
    print(" PROJET TERMINÉ AVEC SUCCÈS!")
    print("="*60)

if __name__ == "__main__":
    main()