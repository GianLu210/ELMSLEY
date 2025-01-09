from sklearn.cluster import KMeans
import numpy as np

def elbow_method(data, max_k=20):
    inertia = []
    for k in range(1, max_k+1):
        kmeans = KMeans(k)
        kmeans.fit(data)
        inertia.append(kmeans.inertia_)

    differences = np.diff(inertia)
    second_differences = np.diff(differences)

    # Trovare il valore di k in corrispondenza del "gomito"
    k_optimal = np.argmin(second_differences) + 2


    return k_optimal