import open3d as o3d
import numpy as np
import pyvista as pv
import matplotlib
#matplotlib.use('TKAgg')
from matplotlib.colors import LinearSegmentedColormap


"""
def read_binary(file, N, isVector=0):
    #file - full path of binary file
    # N - number of vertices
    # isVector - if the data in the binary file is a vector 1, matrix 0

    with open(file, "rb") as f:
        raw_data = np.fromfile(f,np.float64)
    nd_array = raw_data[2:]
    if not isVector:
        M = int(nd_array.shape[0]/N)
        nd_array = nd_array.reshape((M, N))
        nd_array = nd_array.transpose()
    return nd_array
"""

def read_binary_matVec(filename, is_vector=False):
    with open(filename, 'rb') as f:
        # Read dimensions
        rows = np.fromfile(f, dtype=np.int64, count=1)[0]
        cols = np.fromfile(f, dtype=np.int64, count=1)[0]
        # Read the matrix data assuming double precision float (float64)
        data = np.fromfile(f, dtype=np.float64)
    
    if is_vector:
        # If it's a vector, reshape accordingly. Assuming vectors are always column vectors.
        # Since Eigen writes column vectors, we expect rows > 1 and cols == 1
        vector = data.reshape((rows, cols)).flatten()  # Flatten to make sure it's 1D
        return vector
    else:
        # It's a matrix, reshape considering column-major order
        # Transpose to convert from column-major (Eigen) to row-major (NumPy)
        matrix = data.reshape((cols, rows)).T
        return matrix

# return mesh dictionary with vertices, faces and colors with open3d
def read_ply_file(file_path):
    mesh = {}
    plydata = o3d.io.read_triangle_mesh(filename=str(file_path))
    vertices = np.asarray(plydata.vertices)
    faces = np.asarray(plydata.triangles)
    try:
        colors = 255 * np.asarray(plydata.vertex_colors) #-> for TriangleMesh not point cloud
    except:
        colors = None
    
    mesh['vertices'] = vertices
    mesh['faces'] = faces
    mesh['colors'] = colors
    return mesh

#--------------------------------------------------
#pyVista - visualization

def create_pyVista_obj(vertices, faces=None, colors=None):
    if faces is not None:
        stack_vec = faces.shape[1] * np.ones((faces.shape[0], 1), dtype=int) #number of edges in every face
        pv_object = pv.PolyData(vertices, np.hstack((stack_vec, faces)))

    else:
        pv_object = pv.PolyData(vertices)

    if colors is not None:
        pv_object['colors'] = colors / 256

    return pv_object

def pv_compare(mesh1, mesh2, title, types, colors=None):
    # 'colors' var has to be a list
    # 'types' var has to be a list of str

    plotter = pv.Plotter(shape=(1, 2))
    plotter.set_background('w')
    plotter.title = title
    
    if colors is None: # not good because in mesh it is numx3 and not numx1 like the color list
        colors = [mesh1['colors'], mesh2['colors']]

    pv_mesh1 = create_pyVista_obj( vertices = mesh1['vertices'], faces = mesh1['faces'], colors = colors[0])
    pv_mesh2 = create_pyVista_obj( vertices = mesh2['vertices'], faces = mesh2['faces'], colors = colors[1])

    plotter.subplot(0, 0)
    plotter.add_title(types[0], color = 'black')
    plotter.add_mesh(pv_mesh1, show_edges=False, scalar_bar_args={'title': types[0], 'fmt':'%10.3f'}) 

    plotter.subplot(0, 1)
    plotter.add_title(types[1], color = 'black')
    plotter.add_mesh(pv_mesh2, show_edges=False, scalar_bar_args={'title': types[1], 'fmt':'%10.3f'}) 

    plotter.link_views()
    plotter.show()

def pv_compare_3(mesh1, mesh2, mesh3, title, types, colors=None):
    # 'colors' var has to be a list
    # 'types' var has to be a list of str

    plotter = pv.Plotter(shape=(1, 3))
    plotter.set_background('w')
    plotter.title = title
    
    if colors is None: # not good because in mesh it is numx3 and not numx1 like the color list
        colors = [mesh1['colors'], mesh2['colors'],mesh3['colors']]

    pv_mesh1 = create_pyVista_obj( vertices = mesh1['vertices'], faces = mesh1['faces'], colors = colors[0])
    pv_mesh2 = create_pyVista_obj( vertices = mesh2['vertices'], faces = mesh2['faces'], colors = colors[1])
    pv_mesh3 = create_pyVista_obj( vertices = mesh3['vertices'], faces = mesh3['faces'], colors = colors[2])

    plotter.subplot(0, 0)
    plotter.add_title(types[0], color = 'black')
    plotter.add_mesh(pv_mesh1, show_edges=False, scalar_bar_args={'title': types[0], 'fmt':'%10.3f'}) 

    plotter.subplot(0, 1)
    plotter.add_title(types[1], color = 'black')
    plotter.add_mesh(pv_mesh2, show_edges=False, scalar_bar_args={'title': types[1], 'fmt':'%10.3f'}) 
    
    plotter.subplot(0, 2)
    plotter.add_title(types[2], color = 'black')
    plotter.add_mesh(pv_mesh3, show_edges=False, scalar_bar_args={'title': types[2], 'fmt':'%10.3f'}) 

    plotter.link_views()
    plotter.show()

def pv_compare_analysis(mesh1, mesh2, title, types, colors=None):
    # 'colors' var has to be a list
    # 'types' var has to be a list of str

    plotter = pv.Plotter(shape=(1, 2))
    plotter.set_background('w')
    plotter.title = title

    # Access the 'summer' colormap
    summer_cmap = matplotlib.cm.get_cmap("summer")

    # Extract colors from specific points - mid green and end yellow
    green = summer_cmap(0.5)[:3]  # Mid-point of the colormap for green
    yellow = summer_cmap(1.0)[:3]  # End of the colormap for yellow
    
    cmap_colors = {
    'red':   [(0.0, 1.0, 1.0),          # White at the start
              (0.35, 1.0, 1.0),         #still white
              (0.5, green[0], green[0]),  # Start of green
              (0.8, yellow[0], yellow[0]),  # Transition to yellow
              (1.0, yellow[0], yellow[0])],  # End with yellow
    'green': [(0.0, 1.0, 1.0),
              (0.35, 1.0, 1.0), 
              (0.5, green[1], green[1]),
              (0.8, yellow[1], yellow[1]),
              (1.0, yellow[1], yellow[1])],
    'blue':  [(0.0, 1.0, 1.0),
              (0.35, 1.0, 1.0), 
              (0.5, green[2], green[2]),
              (0.8, yellow[2], yellow[2]),
              (1.0, yellow[2], yellow[2])]
}

    # Create a custom LinearSegmentedColormap
    custom_cmap = LinearSegmentedColormap('Custom_White_to_Summer', cmap_colors)

    if colors is None: # not good because in mesh it is numx3 and not numx1 like the color list
        colors = [mesh1['colors'], mesh2['colors']]

    pv_mesh1 = create_pyVista_obj( vertices = mesh1['vertices'], faces = mesh1['faces'], colors = colors[0])
    pv_mesh2 = create_pyVista_obj( vertices = mesh2['vertices'], faces = mesh2['faces'], colors = colors[1])
    center1 = mesh1['vertices'].mean(axis=0)
    center2 = mesh2['vertices'].mean(axis=0)
    # Calculate the center point between two mesh centers
    center_point = (center1 +center2) / 2

    plotter.subplot(0, 0)
    plotter.add_title(types[0], color = 'black')
    plotter.add_mesh(pv_mesh1, show_edges=False, scalar_bar_args={'title': types[0], 'fmt':'%10.3f'}, cmap=custom_cmap,show_scalar_bar=True) 
    # Set this as the focal point for all linked views
    plotter.camera.focal_point = center_point

    plotter.subplot(0, 1)
    plotter.add_title(types[1], color = 'black')
    plotter.add_mesh(pv_mesh2, show_edges=False, scalar_bar_args={'title': types[1], 'fmt':'%10.3f'}, cmap=custom_cmap) 
    # Set this as the focal point for all linked views
    plotter.camera.focal_point = center_point

    plotter.link_views()
    plotter.show()
    return custom_cmap


def add_to_plotter(point_data,plotter):
    # Loop through the parsed data and add each to the plot
    for point_type, data in point_data.items():
        if data['points']:
            # Create a PolyData object for the points
            point_cloud = pv.PolyData(np.array(data['points']))
            # Add the points to the plotter with the specific color
            plotter.add_mesh(point_cloud, color=data['color'], point_size=10, render_points_as_spheres=True)

def add_comma_and_number(file_path, output_file_path=None):
    # If no output file path is given, overwrite the original file
    if output_file_path is None:
        output_file_path = file_path

    # Read all lines from the original file
    with open(file_path, 'r') as file:
        lines = file.readlines()

    # Modify each line to add ',1'
    modified_lines = [line.rstrip('\n') + ',1\n' for line in lines]

    # Write the modified lines back to the output file
    with open(output_file_path, 'w') as file:
        file.writelines(modified_lines)

def extract_points(file_path):
    point_data = {
        'found': {'points': [], 'line_numbers': [], 'color': 'red'},
        'not_found': {'points': [], 'line_numbers': [], 'color': 'black'}
    }

    # Read the file and parse the data
    with open(file_path, 'r') as file:
        for index, line in enumerate(file, start=1):
            parts = line.strip().split(',')
            point_type = parts[0]  # No longer needed if type is irrelevant
            coordinates = list(map(float, parts[1:-1]))  # Assumes last part is found status
            is_found = int(parts[-1])  # Last item is '1' (found) or '0' (not found)

            # Decide which category to put the point into
            category = 'found' if is_found == 1 else 'not_found'
            point_data[category]['points'].append(coordinates)
            point_data[category]['line_numbers'].append(index)

    return point_data

def pv_compare_analysis_coordinates(mesh1, mesh2, title, types, colors=None,file_path=None):
    # 'colors' var has to be a list
    # 'types' var has to be a list of str

    plotter = pv.Plotter(shape=(1, 2))
    plotter.set_background('w')
    plotter.title = title

    point_data = extract_points(file_path)

    # Access the 'summer' colormap
    summer_cmap = matplotlib.cm.get_cmap("summer")

    # Extract colors from specific points - mid green and end yellow
    green = summer_cmap(0.5)[:3]  # Mid-point of the colormap for green
    yellow = summer_cmap(1.0)[:3]  # End of the colormap for yellow
    
    cmap_colors = {
    'red':   [(0.0, 1.0, 1.0),          # White at the start
              (0.35, 1.0, 1.0),         #still white
              (0.5, green[0], green[0]),  # Start of green
              (0.8, yellow[0], yellow[0]),  # Transition to yellow
              (1.0, yellow[0], yellow[0])],  # End with yellow
    'green': [(0.0, 1.0, 1.0),
              (0.35, 1.0, 1.0), 
              (0.5, green[1], green[1]),
              (0.8, yellow[1], yellow[1]),
              (1.0, yellow[1], yellow[1])],
    'blue':  [(0.0, 1.0, 1.0),
              (0.35, 1.0, 1.0), 
              (0.5, green[2], green[2]),
              (0.8, yellow[2], yellow[2]),
              (1.0, yellow[2], yellow[2])]
}

    # Create a custom LinearSegmentedColormap
    custom_cmap = LinearSegmentedColormap('Custom_White_to_Summer', cmap_colors)
    #custom_cmap = summer_cmap

    if colors is None: # not good because in mesh it is numx3 and not numx1 like the color list
        colors = [mesh1['colors'], mesh2['colors']]

    pv_mesh1 = create_pyVista_obj( vertices = mesh1['vertices'], faces = mesh1['faces'], colors = colors[0])
    pv_mesh2 = create_pyVista_obj( vertices = mesh2['vertices'], faces = mesh2['faces'], colors = colors[1])

    plotter.subplot(0, 0)
    plotter.add_title(types[0], color = 'black')
    plotter.add_mesh(pv_mesh1, show_edges=False, scalar_bar_args={'title': types[0], 'fmt':'%10.3f'}, cmap=custom_cmap,show_scalar_bar=True) 
    #add_to_plotter
    for point_type, data in point_data.items():
        if data['points']:
            # Create a PolyData object for the points
            point_cloud = pv.PolyData(np.array(data['points']))
            # Add the points to the plotter with the specific color
            plotter.add_mesh(point_cloud, color=data['color'], point_size=30, render_points_as_spheres=True)

    plotter.subplot(0, 1)
    plotter.add_title(types[1], color = 'black')
    plotter.add_mesh(pv_mesh2, show_edges=False, scalar_bar_args={'title': types[1], 'fmt':'%10.3f'}, cmap=custom_cmap) 
    #add_to_plotter
    for point_type, data in point_data.items():
        if data['points']:
            # Create a PolyData object for the points
            point_cloud = pv.PolyData(np.array(data['points']))
            # Add the points to the plotter with the specific color
            plotter.add_mesh(point_cloud, color=data['color'], point_size=30, render_points_as_spheres=True)

    plotter.link_views()
    plotter.show()




def pv_compare_analysis_coordinates_num(mesh1, mesh2, title, types, colors=None,file_path=None, max_points=None):
    # 'colors' var has to be a list
    # 'types' var has to be a list of str

    plotter = pv.Plotter(shape=(1, 2))
    plotter.set_background('w')
    plotter.title = title

    #add_comma_and_number(file_path)
    point_data = extract_points(file_path)

    # Access the 'summer' colormap
    summer_cmap = matplotlib.cm.get_cmap("summer")

    # Extract colors from specific points - mid green and end yellow
    green = summer_cmap(0.5)[:3]  # Mid-point of the colormap for green
    yellow = summer_cmap(1.0)[:3]  # End of the colormap for yellow
    
    cmap_colors = {
    'red':   [(0.0, 1.0, 1.0),          # White at the start
              (0.35, 1.0, 1.0),         #still white
              (0.5, green[0], green[0]),  # Start of green
              (0.8, yellow[0], yellow[0]),  # Transition to yellow
              (1.0, yellow[0], yellow[0])],  # End with yellow
    'green': [(0.0, 1.0, 1.0),
              (0.35, 1.0, 1.0), 
              (0.5, green[1], green[1]),
              (0.8, yellow[1], yellow[1]),
              (1.0, yellow[1], yellow[1])],
    'blue':  [(0.0, 1.0, 1.0),
              (0.35, 1.0, 1.0), 
              (0.5, green[2], green[2]),
              (0.8, yellow[2], yellow[2]),
              (1.0, yellow[2], yellow[2])]
}

    # Create a custom LinearSegmentedColormap
    custom_cmap = LinearSegmentedColormap('Custom_White_to_Summer', cmap_colors)
    #custom_cmap = summer_cmap

    if colors is None: # not good because in mesh it is numx3 and not numx1 like the color list
        colors = [mesh1['colors'], mesh2['colors']]

    pv_mesh1 = create_pyVista_obj( vertices = mesh1['vertices'], faces = mesh1['faces'], colors = colors[0])
    pv_mesh2 = create_pyVista_obj( vertices = mesh2['vertices'], faces = mesh2['faces'], colors = colors[1])
    
    plotter.subplot(0, 0)
    plotter.add_title(types[0], color = 'black')
    plotter.add_mesh(pv_mesh1, show_edges=False, scalar_bar_args={'title': types[0], 'fmt':'%10.3f'}, cmap=custom_cmap,show_scalar_bar=True) 
    #add_to_plotter
    for point_type, data in point_data.items():
        if data['points']:
            # Create a PolyData object for the points
            point_cloud = pv.PolyData(np.array(data['points']))
            # Add the points to the plotter with the specific color
            plotter.add_mesh(point_cloud, color=data['color'], point_size=50, render_points_as_spheres=True)
            for k, point in enumerate(data['points']): 
                plotter.add_point_labels([point], [str(data['line_numbers'][k])], font_size=40, point_color='red', text_color='red')

    plotter.subplot(0, 1)
    plotter.add_title(types[1], color = 'black')
    plotter.add_mesh(pv_mesh2, show_edges=False, scalar_bar_args={'title': types[1], 'fmt':'%10.3f'}, cmap=custom_cmap) 
    #add_to_plotter
    for point_type, data in point_data.items():
        if data['points']:
            # Create a PolyData object for the points
            point_cloud = pv.PolyData(np.array(data['points']))
            # Add the points to the plotter with the specific color
            plotter.add_mesh(point_cloud, color=data['color'], point_size=50, render_points_as_spheres=True)
            for k, point in enumerate(data['points']): 
                plotter.add_point_labels([point], [str(data['line_numbers'][k])], font_size=40, point_color='red', text_color='red')

    plotter.link_views()
    plotter.show()


def pv_compare_analysis_k_num(mesh1, mesh2, title, types, colors=None,file_path=None, data_max_points=None):
    # 'colors' var has to be a list
    # 'types' var has to be a list of str

    plotter = pv.Plotter(shape=(1, 2))
    plotter.set_background('w')
    plotter.title = title

    #add_comma_and_number(file_path)
    point_data = extract_points(file_path)

    # Access the 'summer' colormap
    summer_cmap = matplotlib.cm.get_cmap("summer")

    # Extract colors from specific points - mid green and end yellow
    green = summer_cmap(0.5)[:3]  # Mid-point of the colormap for green
    yellow = summer_cmap(1.0)[:3]  # End of the colormap for yellow
    
    cmap_colors = {
    'red':   [(0.0, 1.0, 1.0),          # White at the start
              (0.35, 1.0, 1.0),         #still white
              (0.5, green[0], green[0]),  # Start of green
              (0.8, yellow[0], yellow[0]),  # Transition to yellow
              (1.0, yellow[0], yellow[0])],  # End with yellow
    'green': [(0.0, 1.0, 1.0),
              (0.35, 1.0, 1.0), 
              (0.5, green[1], green[1]),
              (0.8, yellow[1], yellow[1]),
              (1.0, yellow[1], yellow[1])],
    'blue':  [(0.0, 1.0, 1.0),
              (0.35, 1.0, 1.0), 
              (0.5, green[2], green[2]),
              (0.8, yellow[2], yellow[2]),
              (1.0, yellow[2], yellow[2])]
}

    # Create a custom LinearSegmentedColormap
    custom_cmap = LinearSegmentedColormap('Custom_White_to_Summer', cmap_colors)

    if colors is None: # not good because in mesh it is numx3 and not numx1 like the color list
        colors = [mesh1['colors'], mesh2['colors']]

    pv_mesh1 = create_pyVista_obj( vertices = mesh1['vertices'], faces = mesh1['faces'], colors = colors[0])
    pv_mesh2 = create_pyVista_obj( vertices = mesh2['vertices'], faces = mesh2['faces'], colors = colors[1])
    
    plotter.subplot(0, 0)
    plotter.add_title(types[0], color = 'black')
    plotter.add_mesh(pv_mesh1, show_edges=False, scalar_bar_args={'title': types[0], 'fmt':'%10.3f'}, cmap=custom_cmap,show_scalar_bar=True) 
    #add_to_plotter
    for point_type, data in point_data.items():
        if data['points']:
            # Create a PolyData object for the points
            point_cloud = pv.PolyData(np.array(data['points']))
            # Add the points to the plotter with the specific color
            plotter.add_mesh(point_cloud, color=data['color'], point_size=50, render_points_as_spheres=True)
            for k, point in enumerate(data['points']): 
                plotter.add_point_labels([point], [str(data['line_numbers'][k])], font_size=40, point_color='red', text_color='red')

    for k, i in enumerate(data_max_points[0]): 
            #plotter.add_point_labels([mesh1['vertices'][i,:]], [str(k)], font_size=40, point_color='blue', text_color='blue')
            plotter.add_point_labels([mesh1['vertices'][i,:]], [str(data_max_points[2][k])], font_size=40, point_color='blue', text_color='blue')

    plotter.subplot(0, 1)
    plotter.add_title(types[1], color = 'black')
    plotter.add_mesh(pv_mesh2, show_edges=False, scalar_bar_args={'title': types[1], 'fmt':'%10.3f'}, cmap=custom_cmap) 
    #add_to_plotter
    for point_type, data in point_data.items():
        if data['points']:
            # Create a PolyData object for the points
            point_cloud = pv.PolyData(np.array(data['points']))
            # Add the points to the plotter with the specific color
            plotter.add_mesh(point_cloud, color=data['color'], point_size=50, render_points_as_spheres=True)
            for k, point in enumerate(data['points']): 
                plotter.add_point_labels([point], [str(data['line_numbers'][k])], font_size=40, point_color='red', text_color='red')
    
    for k, i in enumerate(data_max_points[1]): 
            #plotter.add_point_labels([mesh2['vertices'][i,:]], [str(k)], font_size=40, point_color='blue', text_color='blue')
            plotter.add_point_labels([mesh2['vertices'][i,:]], [str(data_max_points[2][k])], font_size=40, point_color='blue', text_color='blue')

    plotter.link_views()
    plotter.show()
    return custom_cmap