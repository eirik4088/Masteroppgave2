import numpy as np
import math
import matplotlib.pyplot as plt
import plotly as px
from mpl_toolkits.mplot3d import Axes3D
from sklearn.metrics.pairwise import euclidean_distances
from scipy.spatial.distance import cdist
from sklearn.decomposition import PCA

def plot_31_points_1d(data, block):
    plt.figure()
    plt.plot(data[:10], np.zeros((10)),label="Cluster 1", c="r", marker="o")
    plt.scatter(data[10:20], np.zeros((10)), label="Cluster 1", c="g", marker="o")
    plt.scatter(data[20:26], np.zeros((6)), label="Cluster 1", c="b", marker="o")
    plt.show(block=block)

def plot_31_points_2d(data, block):
    plt.figure()
    plt.scatter(data[:10, 0], data[:10, 1], label="Cluster 1", c="r", marker="o")
    plt.scatter(data[10:20, 0], data[10:20, 1], label="Cluster 1", c="g", marker="o")
    plt.scatter(data[20:26, 0], data[20:26, 1], label="Cluster 1", c="b", marker="o")
    plt.show(block=block)

def plot_31_points_3d(data):
    fig = px.graph_objects.Figure()
    fig.add_scatter3d(x=data[:10, 0], y=data[:10, 1], z=data[:10, 2], mode='markers', hovertext=np.char.mod('%d', np.arange(0,10)))
    fig.add_scatter3d(x=data[10:20, 0], y=data[10:20, 1], z=data[10:20, 2], mode='markers', hovertext=np.char.mod('%d', np.arange(10,20)))
    fig.add_scatter3d(x=data[20:26, 0], y=data[20:26, 1], z=data[20:26, 2], mode='markers', hovertext=np.char.mod('%d', np.arange(20,31)))
    # Customize the appearance, labels, and title as needed
    fig.update_traces(marker=dict(size=2))  # Adjust marker size
    fig.update_layout(scene=dict(aspectmode='cube'))  # Equal aspect ratio
    fig.update_layout(scene=dict(xaxis_title='X-axis', yaxis_title='Y-axis', zaxis_title='Z-axis'))
    fig.update_layout(title='Interactive 3D Scatter Plot')
    # Display the interactive plot in a web browser
    fig.show()



def unit_normalize(data: np.ndarray) -> np.ndarray:
    """_summary_

    _extended_summary_  shape=(n_samples, n_features)

    Parameters
    ----------
    data : np.ndarray
        _description_

    Returns
    -------
    np.ndarray
        _description_
    """
    norms = np.linalg.norm(data, axis=1, keepdims=True)
    normalized = data / norms
    return normalized


def feature_transform(data: np.ndarray) -> np.ndarray:
    #print(data)
    """_summary_

    _extended_summary_ shape = (n_samples, n_features)

    Parameters
    ----------
    data : np.ndarray
        _description_

    Returns
    -------
    np.ndarray
        _description_
    """

    ###########################
    #Handling the zero cases later...

    ###########################

    normalized = unit_normalize(data)
    #plot_31_points_2d(normalized, block=False)
    plot_31_points_3d(normalized)

    shift_positive = np.abs(normalized)
    #plot_31_points_2d(shift_positive, block=False)
    plot_31_points_3d(shift_positive)

    transformed = np.ndarray(shape=(data.shape[0], data.shape[1]*2))
    for q in range(shift_positive.shape[1]):
        print(shift_positive)
        radius_vals = np.ndarray(shape=(shift_positive.shape[0], shift_positive.shape[1]-1))
        original_dim = shift_positive.copy()

        for dim in range(shift_positive.shape[1]-1):
            reduced_dim = np.ndarray(shape=(original_dim.shape[0], original_dim.shape[1]-1))
            radius_vals[:, -dim-1] = np.linalg.norm(original_dim, axis=1)
            #print(radius_vals[:, -dim-1])

            for samp in range(shift_positive.shape[0]):
                reduced_vector = np.delete(original_dim[samp, :].copy(), -1)
                reduced_dim[samp, :] = reduced_vector*np.linalg.norm(reduced_vector)*(1/radius_vals[samp, -dim-1])
                
            original_dim = reduced_dim.copy()
            if dim == 0:
                plot_31_points_2d(reduced_dim, block = False)
        
        
        
        plot_31_points_1d(reduced_dim, block=False)
        #print(reduced_dim, radius_vals[:, 0])
        centralize = reduced_dim.ravel() - (radius_vals[:, 0]/2)
        plot_31_points_1d(centralize, block=False)
        #print(radius_vals[:, 0]/2)
        #print(reduced_dim)
        transformed[:, (q*2)] = centralize

        if q == shift_positive.shape[1]-1:
            pull_direction = (data[:, 0] * data[:, -1])/np.abs(data[:, 0] * data[:, -1])
        else:
            pull_direction = (data[:, q] * data[:, q+1])/np.abs(data[:, q] * data[:, q+1])
        np.nan_to_num(pull_direction, copy=False)

        radius_this_dim = np.square(radius_vals[:, 0]/2)
        print(radius_this_dim)
        #radius_this_dim = np.linalg.norm(shift_positive[:, 0:dim+2], axis=1)
        #radius_this_dim = np.square(radius_this_dim/2)

        #print(radius_vals)

        sum_squared_old_dims = -np.square(transformed[:, q*2])


        transformed[:, (q*2)+1] = np.sqrt((sum_squared_old_dims+radius_this_dim))*pull_direction

        """permutation = [1, 2, 0]
        idx = np.empty_like(permutation)
        idx[permutation] = np.arange(len(permutation))"""
        shift_positive = shift_positive[:, [1, 2, 0]]


#########old
    """     
    radius_vals = np.linalg.norm(shift_positive[:, dim:dim+2], axis=1)
        if dim == 1:
            radius_vals = np.ones(31)

        x_vals = shift_positive[:, dim]
        y_vals = shift_positive[:, dim+1]
        linearize = np.ndarray(x_vals.shape)
        for e in range(linearize.shape[0]):
            if x_vals[e] >= y_vals[e]:
                linearize[e] = np.square(x_vals[e])*(1/radius_vals[e])
            else:
                linearize[e] = radius_vals[e] - (np.square(y_vals[e])*(1/radius_vals[e]))
        plot_31_points_1d(linearize, block=False)

        centralize = linearize - (radius_vals/2)
        plot_31_points_1d(centralize, block=False)

        radius_vals = np.square(radius_vals/2)"""
###########
    """sum_squared_old_dims = np.zeros(shape=(transformed.shape[0]))
    
    for dim in range(transformed.shape[1]-1):
        pull_direction = (data[:, dim] * data[:, dim+1])/np.abs(data[:, dim] * data[:, dim+1])
        np.nan_to_num(pull_direction, copy=False)

        radius_this_dim = np.square(radius_vals[:, dim]/2)
        #radius_this_dim = np.linalg.norm(shift_positive[:, 0:dim+2], axis=1)
        #radius_this_dim = np.square(radius_this_dim/2)

        #print(radius_vals)

        sum_squared_old_dims = sum_squared_old_dims - np.square(transformed[:, dim])


        transformed[:, dim+1] = np.sqrt((sum_squared_old_dims+radius_this_dim))*pull_direction"""

    #print(transformed)
    #plot_31_points_2d(transformed, block=True)
    #plot_31_points_3d(transformed)
    #plot_31_points_1d(np.arange(0, 31), block=True)

    #print(transformed)
    plot_31_points_2d(transformed[:, 0:2], block=False)
    plot_31_points_2d(transformed[:, 2:4], block=False)
    plot_31_points_2d(transformed[:, 4:6], block=True)
    #print(transformed)
    return transformed

def make_order_vector(vector: np.array) -> np.array:
    sort_index = np.argsort(vector, kind='mergesort')
    return sort_index

def asses_metric_order(distance_func, data: np.ndarray):
    transformed = feature_transform(data)
    normalized = unit_normalize(data)
    absolute_cosine_similarity = np.abs(normalized.dot(normalized.T))
    absolute_cosine_distance = np.abs(np.around(absolute_cosine_similarity, 10) - 1)
    test_distance = distance_func(transformed, transformed)
    test_distance = np.around(test_distance, 10)

    consistent_metric_order = True

    for obsv in range(len(data)):
        #if make_order_vector(absolute_cosine_distance[:, obsv]) != make_order_vector(test_distance[:, obsv]):
        #print(make_order_vector(absolute_cosine_distance[:, obsv]))
        #print(absolute_cosine_distance[8, [9, 18,  0,  7,  3,  4, 20, 21, 23, 24,  8, 10, 14, 16, 17, 19, 11, 13, 22,  1,  2,  5,  6, 25, 12, 15]])
        #print(make_order_vector(test_distance[:, obsv]))
        #print(test_distance[obsv, [9, 18,  0,  7,  3,  4, 20, 21, 23, 24,  8, 10, 14, 16, 17, 19, 11, 13, 22,  1,  2,  5,  6, 25, 12, 15]])
        if not all(v == 0 for v in (make_order_vector(absolute_cosine_distance[:, obsv]) - make_order_vector(test_distance[:, obsv]))):
            consistent_metric_order = False
    print(make_order_vector(absolute_cosine_distance[9, :]))
    print(make_order_vector(test_distance[9, :]))
    print(absolute_cosine_distance[9, [9, 18,  0,  7,  3,  4, 20, 21, 23, 24,  8, 10, 14, 16, 17, 19, 11, 13, 22,  1,  2,  5,  6, 25, 12, 15]])
    print(test_distance[9, [9, 18,  0,  7,  3,  4, 20, 21, 23, 24,  8, 10, 14, 16, 17, 19, 11, 13, 22,  1,  2,  5,  6, 25, 12, 15]])
    #print(all(v == 0 for v in (make_order_vector(absolute_cosine_distance[:, 0]) - make_order_vector(test_distance[:, 0]))))
    return consistent_metric_order


"""
# Number of points to generate
num_points = 31

# Generate evenly spaced angles to cover the unit circle
angles = np.linspace(0, 2 * np.pi, num_points, endpoint=False)

# Convert angles to 2D vectors
data_2d = np.array([(np.cos(theta), np.sin(theta)) for theta in angles])


print(asses_metric_order(cdist, data_2d))


data = feature_transform(data_2d)
            #np.hstack((np.abs(data_all), (((data_all[:, 0] * data_all[:, 1])/np.abs((data_all[:, 0] * data_all[:, 1])))*np.sqrt(data_all[:, 0] + data_all[:, 1] - 1)).reshape((len(data_all), 1))))

"""
def fibonacci_sphere(samples=100):

    points = np.ndarray(shape=(31, 3))
    phi = math.pi * (3. - math.sqrt(5.))  # golden angle in radians

    for i in range(samples):
        y = 1 - (i / float(samples - 1)) * 2  # y goes from 1 to -1
        radius = math.sqrt(1 - y * y)  # radius at y

        theta = phi * i  # golden angle increment

        x = math.cos(theta) * radius
        z = math.sin(theta) * radius

        points[i, 0] = x
        points[i, 1] = y
        points[i, 2] = z

    return points

# Convert the list of points to a NumPy array
data_3d = fibonacci_sphere(31)

data_3d = np.where(data_3d==0, 0.00001, data_3d)

l = [[np.sqrt(0.33333), np.sqrt(0.33333), np.sqrt(0.33333)],
     [-np.sqrt(0.33333), np.sqrt(0.33333), np.sqrt(0.33333)],
     [np.sqrt(0.33333), -np.sqrt(0.33333), np.sqrt(0.33333)],
     [np.sqrt(0.33333), np.sqrt(0.33333), -np.sqrt(0.33333)],
     [-np.sqrt(0.33333), -np.sqrt(0.33333), np.sqrt(0.33333)],
     [np.sqrt(0.33333), -np.sqrt(0.33333), -np.sqrt(0.33333)],
     [-np.sqrt(0.33333), np.sqrt(0.33333), -np.sqrt(0.33333)],
     [-np.sqrt(0.33333), -np.sqrt(0.33333), -np.sqrt(0.33333)],
     [np.sqrt(0.5), 0, np.sqrt(0.5)],
     [np.sqrt(0.5), np.sqrt(0.5), 0],
     [0, np.sqrt(0.5), np.sqrt(0.5)],
     [-np.sqrt(0.5), 0, np.sqrt(0.5)],
     [-np.sqrt(0.5), np.sqrt(0.5), 0],
     [0, -np.sqrt(0.5), np.sqrt(0.5)],
     [np.sqrt(0.5), 0, -np.sqrt(0.5)],
     [np.sqrt(0.5), -np.sqrt(0.5), 0],
     [0, np.sqrt(0.5), -np.sqrt(0.5)],
     [-np.sqrt(0.5), 0, -np.sqrt(0.5)],
     [-np.sqrt(0.5), -np.sqrt(0.5), 0],
     [0, -np.sqrt(0.5), -np.sqrt(0.5)],
     [1, 0, 0],
     [0, 1, 0],
     [0, 0, 1],
     [-1, 0, 0],
     [0, -1, 0],
     [0, 0, -1]]

data_3d = np.array(l)
data_3d = np.where(data_3d==0, 0.00001, data_3d)




#print(data_3d)
#plot_31_points_3d(data_3d)
print(asses_metric_order(cdist, data_3d))