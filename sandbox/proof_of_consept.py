import numpy as np
import math
import matplotlib.pyplot as plt
import plotly as px
from mpl_toolkits.mplot3d import Axes3D
from sklearn.metrics.pairwise import euclidean_distances
from scipy.spatial.distance import cdist

def plot_31_points_1d(data, block):
    plt.figure()
    plt.plot(data[:10], np.zeros((10)),label="Cluster 1", c="r", marker="o")
    plt.scatter(data[10:20], np.zeros((10)), label="Cluster 1", c="g", marker="o")
    plt.scatter(data[20:31], np.zeros((11)), label="Cluster 1", c="b", marker="o")
    plt.show(block=block)

def plot_31_points_2d(data, block):
    plt.figure()
    plt.scatter(data[:10, 0], data[:10, 1], label="Cluster 1", c="r", marker="o")
    plt.scatter(data[10:20, 0], data[10:20, 1], label="Cluster 1", c="g", marker="o")
    plt.scatter(data[20:31, 0], data[20:31, 1], label="Cluster 1", c="b", marker="o")
    plt.show(block=block)

def plot_31_points_3d(data):
    fig = px.graph_objects.Figure()
    fig.add_scatter3d(x=data[:10, 0], y=data[:10, 1], z=data[:10, 2], mode='markers')
    fig.add_scatter3d(x=data[10:20, 0], y=data[10:20, 1], z=data[10:20, 2], mode='markers')
    fig.add_scatter3d(x=data[20:31, 0], y=data[20:31, 1], z=data[20:31, 2], mode='markers')
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
    normalized = unit_normalize(data)
    #plot_31_points_2d(normalized, block=False)
    plot_31_points_3d(normalized)

    shift_positive = np.abs(normalized)
    #plot_31_points_2d(shift_positive, block=False)
    plot_31_points_3d(shift_positive)

    """linearize = 0

    centralize = shift_positive - 0.5
    plot_31_points_2d(centralize, block=False)"""

    transformed = np.ndarray(shape=data.shape)
    #transformed[:, -1] = shift_positive[:, -1]

    for dim in range(transformed.shape[1]-1):
        pull_direction = (data[:, dim] * data[:, dim+1])/np.abs(data[:, dim] * data[:, dim+1])
        np.nan_to_num(pull_direction, copy=False)

        """linearized_x = ((shift_positive[:, dim]-shift_positive[:, dim+1]+1)/2).reshape((len(shift_positive), 1))
        linearize = np.hstack((linearized_x, -linearized_x+1))
        plot_31_points_2d(linearize, block=False)

        reduced_dim = linearized_x[:, 0]
        plot_31_points_1d(reduced_dim, block=False)"""
        x_vals = shift_positive[:, dim]
        y_vals = shift_positive[:, dim+1]
        linearize = np.ndarray(x_vals.shape)
        for e in range(linearize.shape[0]):
            if x_vals[e] >= y_vals[e]:
                linearize[e] = np.square(x_vals[e])
            else:
                linearize[e] = 1 - np.square(y_vals[e]) 
        plot_31_points_1d(linearize, block=False)

        centralize = linearize - 0.5
        plot_31_points_1d(centralize, block=False)

        if dim == 0:
            transformed[:, dim] = centralize

        transformed[:, dim+1] = np.sqrt((-np.square(centralize)+0.25))*pull_direction

    #print(transformed)
    #plot_31_points_2d(transformed, block=True)
    plot_31_points_3d(transformed)
    plot_31_points_1d(np.arange(0, 31), block=True)
    """dim_reduced = shift_positive[:-2, :]
    plot_31_points_1d(dim_reduced)

    pull_direction = (data * data)/np.abs(data * data)
    print(pull_direction)"""


    """
    new_dim_values = np.ndarray(shape=(data.shape[0], data.shape[1]-1))
    for dim in range(data.shape[1]-1):
        new_dim_values[: , dim] = ((data[:, dim] * data[:, dim+1])/np.abs((data[:, dim] * data[:, dim+1])))*(original_dim_values[:, dim]**1.2)*(original_dim_values[:, dim+1]**1.3)
    np.nan_to_num(new_dim_values, copy=False)
    transformed = np.hstack((original_dim_values, new_dim_values))
    print(transformed)"""
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
        #print(make_order_vector(test_distance[:, obsv]))
        if not all(v == 0 for v in (make_order_vector(absolute_cosine_distance[:, obsv]) - make_order_vector(test_distance[:, obsv]))):
            consistent_metric_order = False

    #print(all(v == 0 for v in (make_order_vector(absolute_cosine_distance[:, 0]) - make_order_vector(test_distance[:, 0]))))
    return consistent_metric_order


# Number of points to generate
num_points = 31

"""# Generate evenly spaced angles to cover the unit circle
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


#print(data_3d)
#plot_31_points_3d(data_3d)
print(asses_metric_order(cdist, data_3d))