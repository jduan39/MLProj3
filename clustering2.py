import csv
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import datetime
import numpy as np
from sklearn import preprocessing, metrics
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA, FastICA
from scipy.stats import kurtosis
from sklearn.utils.extmath import safe_sparse_dot
from sklearn.mixture import GaussianMixture
from sklearn.neural_network import MLPClassifier
from sklearn.model_selection import learning_curve
from sklearn.random_projection import GaussianRandomProjection
from sklearn.feature_selection import SelectKBest, chi2
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.metrics.cluster import normalized_mutual_info_score
from sklearn.metrics import silhouette_samples, silhouette_score, calinski_harabaz_score, homogeneity_completeness_v_measure


df = pd.read_csv('inpatientCharges.csv')
df = df.drop(['Provider Id', 'Provider Name', 'Provider Street Address', 'Provider Zip Code', 'Hospital Referral Region Description'
              , ' Total Discharges ', ' Average Covered Charges ', 'Average Medicare Payments'], axis=1)
df = df.iloc[0:2000]


def preprocess_hospital_df(df):
    processed_df = df.copy()
    le = preprocessing.LabelEncoder()
    processed_df['DRG Definition'] = le.fit_transform(processed_df['DRG Definition'])
    processed_df['Provider City'] = le.fit_transform(processed_df['Provider City'])
    processed_df['Provider State'] = le.fit_transform(processed_df['Provider State'])
    processed_df[' Average Total Payments '] = processed_df[' Average Total Payments '].map(lambda x: x.lstrip('$'))

    return processed_df

processeddf = preprocess_hospital_df(df)
X = processeddf.iloc[:, 0:3]
Y = processeddf.iloc[:, 3]




scaler = MinMaxScaler()
X_scaled = scaler.fit_transform(X)

X_norm = pd.DataFrame(X_scaled)


# K means clustering hospital
# for n_clusters in range(2,11):
#     # Create a subplot with 1 row and 2 columns
#     fig, (ax1, ax2) = plt.subplots(1, 2)
#     fig.set_size_inches(18, 7)
#
#     # The 1st subplot is the silhouette plot
#     # The silhouette coefficient can range from -1, 1 but in this example all
#     # lie within [-0.1, 1]
#     ax1.set_xlim([-0.1, 1])
#     # The (n_clusters+1)*10 is for inserting blank space between silhouette
#     # plots of individual clusters, to demarcate them clearly.
#     ax1.set_ylim([0, len(X_norm) + (n_clusters + 1) * 10])
#
#     # Initialize the clusterer with n_clusters value and a random generator
#     # seed of 10 for reproducibility.
#     clusterer = KMeans(n_clusters=n_clusters, random_state=10).fit(X_norm)
#     cluster_labels = clusterer.labels_
#     print("NMI score: %.6f" % normalized_mutual_info_score(Y, cluster_labels))
#
#     # The silhouette_score gives the average value for all the samples.
#     # This gives a perspective into the density and separation of the formed
#     # clusters
#     silhouette_avg = silhouette_score(X_norm, cluster_labels)
#     print("For n_clusters =", n_clusters,
#           "The average silhouette_score is :", silhouette_avg)
#
#     calinski_score = calinski_harabaz_score(X_norm, cluster_labels)
#
#     print(calinski_score)
#
#     homogeneity, completeness, vmeasure = homogeneity_completeness_v_measure(Y, cluster_labels)
#
#     print(homogeneity)
#     print(completeness)
#
#     # Compute the silhouette scores for each sample
#     sample_silhouette_values = silhouette_samples(X_norm, cluster_labels)
#
#     y_lower = 10
#     for i in range(n_clusters):
#         # Aggregate the silhouette scores for samples belonging to
#         # cluster i, and sort them
#         ith_cluster_silhouette_values = \
#             sample_silhouette_values[cluster_labels == i]
#
#         ith_cluster_silhouette_values.sort()
#
#         size_cluster_i = ith_cluster_silhouette_values.shape[0]
#         y_upper = y_lower + size_cluster_i
#
#
#         cmap = cm.get_cmap("Spectral")
#         color = cmap(float(i)/ n_clusters)
#         # color = cm.spectral(float(i) / n_clusters)
#         ax1.fill_betweenx(np.arange(y_lower, y_upper),
#                           0, ith_cluster_silhouette_values,
#                           facecolor=color, edgecolor=color, alpha=0.7)
#
#         # Label the silhouette plots with their cluster numbers at the middle
#         ax1.text(-0.05, y_lower + 0.5 * size_cluster_i, str(i))
#
#         # Compute the new y_lower for next plot
#         y_lower = y_upper + 10  # 10 for the 0 samples
#
#     ax1.set_title("The silhouette plot for the various clusters.")
#     ax1.set_xlabel("The silhouette coefficient values")
#     ax1.set_ylabel("Cluster label")
#
#     # The vertical line for average silhouette score of all the values
#     ax1.axvline(x=silhouette_avg, color="red", linestyle="--")
#
#     ax1.set_yticks([])  # Clear the yaxis labels / ticks
#     ax1.set_xticks([-0.1, 0, 0.2, 0.4, 0.6, 0.8, 1])
#
#     # 2nd Plot showing the actual clusters formed
#     cmap = cm.get_cmap("Spectral")
#     colors = cmap(cluster_labels.astype(float) / n_clusters)
#     ax2.scatter(X_norm.iloc[:, 1], X_norm.iloc[:, 2], marker='.', s=30, lw=0, alpha=0.7,
#                 c=colors, edgecolor='k')
#
#     # Labeling the clusters
#     centers = clusterer.cluster_centers_
#
#     # Draw white circles at cluster centers
#     ax2.scatter(centers[:, 1], centers[:, 2], marker='o',
#                 c="white", alpha=1, s=200, edgecolor='k')
#
#     for i, c in enumerate(centers):
#         ax2.scatter(c[1], c[2], marker='$%d$' % i, alpha=1,
#                     s=50, edgecolor='k')
#
#     ax2.set_title("The visualization of the clustered data.")
#     ax2.set_xlabel("Feature space for the 1st feature")
#     ax2.set_ylabel("Feature space for the 2nd feature")
#
#     plt.suptitle(("Silhouette analysis for KMeans clustering on sample data "
#                   "with n_clusters = %d" % n_clusters),
#                  fontsize=14, fontweight='bold')
#
#     plt.show()

# Expectation Maximization hospital
# for n_clusters in range(2,11):
#     fig = plt.gcf()
#     fig.set_size_inches(7, 7)
#     ax = fig.add_subplot(111)
#
#     # Initialize the clusterer with n_clusters value and a random generator
#     # seed of 10 for reproducibility.
#     clusterer = GaussianMixture(n_components=n_clusters).fit(X_norm)
#     cluster_labels = clusterer.predict(X_norm)
#     print("For n clusters =", n_clusters)
#     # print("NMI score: %.6f" % normalized_mutual_info_score(Y, cluster_labels))
#     calinski_score = calinski_harabaz_score(X_norm, cluster_labels)
#     print(calinski_score)
#
#     homogeneity, completeness, vmeasure = homogeneity_completeness_v_measure(Y, cluster_labels)
#
#     print(homogeneity)
#     print(completeness)
#
#     # 2nd Plot showing the actual clusters formed
#     cmap = cm.get_cmap("Spectral")
#     colors = cmap(cluster_labels.astype(float) / n_clusters)
#     plt.scatter(X_norm.iloc[:, 1], X_norm.iloc[:, 2], marker='.', s=30, lw=0, alpha=0.7,
#                 c=colors, edgecolor='k')
#
#     # Labeling the clusters
#     centers = clusterer.means_
#
#     # Draw white circles at cluster centers
#     plt.scatter(centers[:, 1], centers[:, 2], marker='o',
#                 c="white", alpha=1, s=200, edgecolor='k')
#
#     for i, c in enumerate(centers):
#         ax.scatter(c[1], c[2], marker='$%d$' % i, alpha=1,
#                    s=50, edgecolor='k')
#
#     ax.set_title("The visualization of the clustered data.")
#     ax.set_xlabel("Feature space for the 1st feature")
#     ax.set_ylabel("Feature space for the 2nd feature")
#
#     plt.suptitle(("Clusters plot for EM clustering on sample data "
#                   "with n_clusters = %d" % n_clusters),
#                  fontsize=14, fontweight='bold')
#
#     plt.show()

# PCA hospital
# for n_components in range(1, 4):
#     pca = PCA(n_components=n_components)
#     X_r = pca.fit(X_norm).transform(X_norm)
#     X_pca = X_r
#     sum = 0
#     for item in pca.explained_variance_ratio_:
#         sum+=item
#     print(sum)

 # ICA hospital
# for n_components in range(1, 4):
#     ica = FastICA(n_components=n_components, random_state=10)
#     X_r = ica.fit(X_norm).transform(X_norm)
#     X_ica = X_r
#     print("N components", n_components)
#     kurt = kurtosis(X_ica)
#     print("Kurtosis",kurt)
    # proj = ica.inverse_transform(X_ica)
    # loss = sum(sum((X_scaled - proj) ** 2))
    # print(loss)
# RP hospital
# print("Loss list:")
# for trial in range(0, 20):
#     rmp = GaussianRandomProjection(n_components=3)
#     X_r = rmp.fit_transform(X_norm)
#     X_rmp = X_r
#     inv_comp = np.linalg.pinv(rmp.components_)
#     recon = safe_sparse_dot(X_rmp, inv_comp.T)
#     loss = sum(sum((X_scaled - recon) ** 2))
#     print(loss)


#KBest titanic
# for k in range(1, 4):
#     print("K =", k)
#     kbest = SelectKBest(chi2, k=k)
#     X_kbest = kbest.fit_transform(X_norm, Y)
#     proj = kbest.inverse_transform(X_kbest)
#     loss = sum(sum((X_scaled - proj) ** 2))
#     print(loss)
#     featuresscores = kbest.scores_
#     for score in featuresscores:
#         print(score)
pca = PCA(n_components=3)
X_pca = pca.fit(X_norm).transform(X_norm)
X_pcatest = pd.DataFrame(X_pca)

# for n_clusters in range(2, 11):
#
#     clusterer = KMeans(n_clusters=n_clusters).fit(X_pcatest)
#     cluster_labels = clusterer.labels_
#     EMclusterer = GaussianMixture(n_components=n_clusters).fit(X_pcatest)
#     EMcluster_labels = EMclusterer.predict(X_pcatest)
#
#
#     silhouette_avg = silhouette_score(X_pcatest, cluster_labels)
#     print("For n_clusters =", n_clusters,
#           "The average silhouette_score is :", silhouette_avg)
#     print("The NMI score is: %.6f" % normalized_mutual_info_score(Y, cluster_labels))
#     calinski_score = calinski_harabaz_score(X_pcatest, cluster_labels)
#
#     print(calinski_score)
#
#     homogeneity, completeness, vmeasure = homogeneity_completeness_v_measure(Y, cluster_labels)
#
#     print(homogeneity)
#     print(completeness)
#
#     emcal = calinski_harabaz_score(X_pcatest, EMcluster_labels)
#     emhomogeneity, emcompleteness, vmeasure = homogeneity_completeness_v_measure(Y, EMcluster_labels)
#     print(emcal)
#     print(emhomogeneity)
#     print(emcompleteness)



ica = FastICA(n_components=3)
X_ica = ica.fit(X_norm).transform(X_norm)
X_icatest = pd.DataFrame(X_ica)

# for n_clusters in range(2, 11):
#
#     clusterer = KMeans(n_clusters=n_clusters).fit(X_icatest)
#     cluster_labels = clusterer.labels_
#     EMclusterer = GaussianMixture(n_components=n_clusters).fit(X_icatest)
#     EMcluster_labels = EMclusterer.predict(X_icatest)
#
#     silhouette_avg = silhouette_score(X_icatest, cluster_labels)
#     print("For n_clusters =", n_clusters,
#           "The average silhouette_score is :", silhouette_avg)
#     print("The NMI score is: %.6f" % normalized_mutual_info_score(Y, cluster_labels))
#     calinski_score = calinski_harabaz_score(X_icatest, cluster_labels)
#
#     print(calinski_score)
#     homogeneity, completeness, vmeasure = homogeneity_completeness_v_measure(Y, cluster_labels)
#
#     print(homogeneity)
#     print(completeness)
#
#     emcal = calinski_harabaz_score(X_icatest, EMcluster_labels)
#     emhomogeneity, emcompleteness, vmeasure = homogeneity_completeness_v_measure(Y, EMcluster_labels)
#     print(emcal)
#     print(emhomogeneity)
#     print(emcompleteness)


rp = GaussianRandomProjection(n_components=3)
X_rp = rp.fit_transform(X_norm)
X_rptest = pd.DataFrame(X_rp)

# for n_clusters in range(2, 11):
#
#     clusterer = KMeans(n_clusters=n_clusters).fit(X_rptest)
#     cluster_labels = clusterer.labels_
#     EMclusterer = GaussianMixture(n_components=n_clusters).fit(X_rptest)
#     EMcluster_labels = EMclusterer.predict(X_rptest)
#
#     silhouette_avg = silhouette_score(X_rptest, cluster_labels)
#     print("For n_clusters =", n_clusters,
#           "The average silhouette_score is :", silhouette_avg)
#     print("The NMI score is: %.6f" % normalized_mutual_info_score(Y, cluster_labels))
#     calinski_score = calinski_harabaz_score(X_rptest, cluster_labels)
#
#     print(calinski_score)
#
#
#     homogeneity, completeness, vmeasure = homogeneity_completeness_v_measure(Y, cluster_labels)
#
#     print(homogeneity)
#     print(completeness)
#
#     emcal = calinski_harabaz_score(X_rptest, EMcluster_labels)
#     emhomogeneity, emcompleteness, vmeasure = homogeneity_completeness_v_measure(Y, EMcluster_labels)
#     print(emcal)
#     print(emhomogeneity)
#     print(emcompleteness)

kbest = SelectKBest(chi2, k=3)
X_kbest = kbest.fit_transform(X_norm, Y)
X_ktest = pd.DataFrame(X_kbest)

for n_clusters in range(2, 11):

    clusterer = KMeans(n_clusters=n_clusters).fit(X_ktest)
    cluster_labels = clusterer.labels_
    EMclusterer = GaussianMixture(n_components=n_clusters).fit(X_ktest)
    EMcluster_labels = EMclusterer.predict(X_ktest)

    silhouette_avg = silhouette_score(X_ktest, cluster_labels)
    print("For n_clusters =", n_clusters,
          "The average silhouette_score is :", silhouette_avg)
    print("The NMI score is: %.6f" % normalized_mutual_info_score(Y, cluster_labels))
    calinski_score = calinski_harabaz_score(X_ktest, cluster_labels)

    print(calinski_score)


    homogeneity, completeness, vmeasure = homogeneity_completeness_v_measure(Y, cluster_labels)

    print(homogeneity)
    print(completeness)

    emcal = calinski_harabaz_score(X_ktest, EMcluster_labels)
    emhomogeneity, emcompleteness, vmeasure = homogeneity_completeness_v_measure(Y, EMcluster_labels)
    print(emcal)
    print(emhomogeneity)
    print(emcompleteness)