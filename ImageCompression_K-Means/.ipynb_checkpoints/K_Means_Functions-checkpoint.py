import numpy as np
import sys

def get_random_centroids(X, k):

    '''
    Each centroid is a point in RGB space (color) in the image. 
    This function should uniformly pick `k` centroids from the dataset.
    Input: a single image of shape `(num_pixels, 3)` and `k`, the number of centroids. 
    Notice we are flattening the image to a two dimentional array.
    Output: Randomly chosen centroids of shape `(k,3)` as a numpy array. 
    '''
    centroids = []
    centriods_index=[]
    segment_length = divmod(X.shape[0], k+1)[0]
    for segment in range(1,k+1):
        centroids.append(X[segment*segment_length,:])
        centriods_index.append(segment*segment_length)
    return np.asarray(centroids).astype(np.float)

def naive_pow(n, k):
    res=n
    for i in range(1,k):
        res *= n
    return res
def lp_distance(X, centroids, p=2):

    '''
    Inputs: 
    A single image of shape (num_pixels, 3)
    The centroids (k, 3)
    The distance parameter p

    output: numpy array of shape `(k, num_pixels)` thats holds the distances of 
    all points in RGB space from all centroids
    '''

    k = len(centroids)
    n = X.shape[1]
    num_pixels=X.shape[0]
    RGB_distance=np.zeros([k,num_pixels])
    for i,c in enumerate(centroids):
        for j,x in enumerate(X):
            dist = 0
            for dim in range(n):
                dist+=naive_pow(np.abs(x[dim] - c[dim]),p)
            RGB_distance[i,j]=dist ** (1 / p)
    return RGB_distance

def kmeans(X, k, p=2 ,max_iter=100, centroids=[]):
    """
    Inputs:
    - X: a single image of shape (num_pixels, 3).
    - k: number of centroids.
    - p: the parameter governing the distance measure.
    - max_iter: the maximum number of iterations to perform.p

    Outputs:
    - The calculated centroids as a numpy array.
    - The final assignment of all RGB points to the closest centroids as a numpy array.
    """
    classes = []
    if centroids == []:
        centroids = get_random_centroids(X, k)
    for iter in range(max_iter):
        print(iter)
        X_dist=lp_distance(X,centroids)
        X_cent=X_dist.argmin(axis=0)
        for i in range(k):
            centroids[i-1]=X[X_dist.argmin(axis=0) == i].mean(axis=0)
        print(centroids)
    classes=X_dist.argmin(axis=0)
    return centroids, classes

# euclidean distance
def distance(p1, p2):
    return np.sum((p1 - p2)**2)

def kmeans_pp(X, k, p ,max_iter=100):
    """
    Your implenentation of the kmeans++ algorithm.
    Inputs:
    - X: a single image of shape (num_pixels, 3).
    - k: number of centroids.
    - p: the parameter governing the distance measure.
    - max_iter: the maximum number of iterations to perform.

    Outputs:
    - The calculated centroids as a numpy array.
    - The final assignment of all RGB points to the closest centroids as a numpy array.
    """
    centroids = []

    # initialize the centroids list & a add random point
    centroids.append(X[np.random.randint(X.shape[0]), :])
    
    # k - 1 centroids loop 
    for j in range(k - 1):
         
        distances = []
        for i in range(X.shape[0]):
            point = X[i, :]
            d = sys.maxsize

            # compute distance of 'point' from each of the previously selected centroid
            for j in range(len(centroids)):
                temp = distance(point, centroids[j])
                d = min(d, temp)
            distances.append(d)
             
        # select data point with maximum distance as centroid
        distances = np.array(distances)
        new_centroid = X[np.argmax(distances), :]
        centroids.append(new_centroid)

    # convert centroids
    list_of_lists = [l.tolist() for l in centroids]
    centroids = list_of_lists
    centroids = np.asarray(centroids).astype(np.float)

    # send to K-means with centroids
    return kmeans(X, k, p ,max_iter, centroids)