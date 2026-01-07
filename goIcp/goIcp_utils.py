import open3d as o3d
import numpy as np
from scipy import ndimage
import copy
import queue as q
import polyscope as ps
import pymeshlab
import pyvista as pv
import sys
import math

def manifold(models, model_type):
    print(f'To manifold process')
    ms = pymeshlab.MeshSet()
    paths = [models['model1_path'], models['model2_path']]
    manifold_paths = [models['manifold_model1_path'], models['manifold_model2_path']]
    log_paths = [models['manifold_log_path1'], models['manifold_log_path2']]
    for i in np.arange(len(paths)): 
        ms.load_new_mesh(paths[i])
        topo_1 = ms.get_topological_measures()
        ms.meshing_repair_non_manifold_edges()
        ms.meshing_repair_non_manifold_vertices()
        ms.meshing_remove_connected_component_by_diameter()
        #check topological measures before and after repair
        topo_2 = ms.get_topological_measures()
        print(f'{model_type}: Check topological measures after the repair in model {i + 1}')
        if topo_2['connected_components_number'] != 1:
            print(f'There is more than 1 connected component in mesh {i+1}')
            return False
        if topo_2['is_mesh_two_manifold'] != True:
            print(f'Mesh {i+1} is not manifold')
            return False
        
        # save topological measures to log (before and after)
        log_path =  log_paths[i]
        with open(log_path, 'w') as f:
            f.write('Before:\n')
            for key, value in topo_1.items(): 
                f.write('%s:%d\n' % (key, value))
            f.write('\nAfter:\n')
            for key, value in topo_2.items(): 
                f.write('%s:%d\n' % (key, value))
          

        #save the repaired mesh - only if it is manifold
        save_path = manifold_paths[i]
        ms.save_current_mesh(file_name=save_path, binary=False, save_vertex_quality=False,
                            save_vertex_radius=False, save_face_quality=False, save_face_color=False,
                            save_wedge_color=False, save_wedge_normal=False)
        # clear the mesh set after saving
        ms.clear()

    models['model1_path'] = models['manifold_model1_path']
    models['model2_path'] = models['manifold_model2_path']
    return True

# return PointCloud and trimesh of open3d
def read_file(file_path, file_type='ply'):
    if file_type == 'ply':
        plydata = o3d.io.read_triangle_mesh(filename=str(file_path))
        vertices = np.array(plydata.vertices)
        #faces = np.asarray(plydata.triangles)
        #try:
        #    colors = 255 * np.asarray(plydata.vertex_colors) -> for TriangleMesh not point cloud
        #except:
        #    colors = None
        pcd = o3d.geometry.PointCloud()
        pcd.points = o3d.utility.Vector3dVector(vertices)
        #return [vertices, faces, colors]

    elif file_type == 'txt':
        pcd = o3d.io.read_point_cloud(file_path, format='xyz')
        #pcd = np.loadtxt(file_path, skiprows=1)
        #plist = pcd.tolist()
    
    return [pcd, plydata] #return point cloud and triangle mesh

#----------------------------------------------------
# As the example in GO-ICP website 
def centralize_and_normalize(mesh1, mesh2, safety_margin=0.95):

    # Convert mesh vertices to NumPy arrays for easier manipulation
    verts1 = copy.deepcopy(np.asarray(mesh1.vertices))
    verts2 = copy.deepcopy(np.asarray(mesh2.vertices))

    # Center to origin - find centroid
    centroid1 = np.mean(verts1, axis=0)
    centroid2 = np.mean(verts2, axis=0)
    
    centralized_verts1 = verts1 - centroid1
    centralized_verts2 = verts2 - centroid2

    print('Finding scale factor')
    # Calculate the maximum absolute distances from the origin for both meshes
    max_dist1 = np.max(np.linalg.norm(centralized_verts1, axis=1))
    max_dist2 = np.max(np.linalg.norm(centralized_verts2, axis=1))
    
    max_dist1 = math.ceil(max_dist1 * 100) / 100
    max_dist2 = math.ceil(max_dist2 * 100) / 100
    
    common_max_dist = max(max_dist1, max_dist2)

    # Scale factor to ensure everything fits within [-1, 1]
    scale_factor = safety_margin / common_max_dist
    print(f'The scale factor is: {scale_factor}')

    return centroid1, centroid2, scale_factor

# As the example in GO-ICP website 
def centralize_and_normalize_GT(mesh1, mesh2, safety_margin=0.9):

    # Convert mesh vertices to NumPy arrays for easier manipulation
    verts1 = copy.deepcopy(np.asarray(mesh1.vertices))
    verts2 = copy.deepcopy(np.asarray(mesh2.vertices))

    # Center to origin - find centroid
    centroid = np.mean(verts1, axis=0)
    
    centralized_verts1 = verts1 - centroid
    centralized_verts2 = verts2 - centroid

    print('Finding scale factor')
    # Calculate the maximum absolute distances from the origin for both meshes
    max_dist1 = np.max(np.linalg.norm(centralized_verts1, axis=1))
    max_dist2 = np.max(np.linalg.norm(centralized_verts2, axis=1))

    max_dist11 = math.ceil(max_dist1 * 100) / 100
    max_dist22 = math.ceil(max_dist2 * 100) / 100

    common_max_dist = max(max_dist11, max_dist22)

    # Scale factor to ensure everything fits within [-1, 1]
    scale_factor = safety_margin / common_max_dist
    print(f'The scale factor is: {scale_factor}')
        
    return centroid, scale_factor

def create_transformation_matrix(centroid, scale_factor=1):
    # Create a 4x4 identity matrix for translation
    T = np.eye(4)
    
    # Set the translation components (negative centroid values for moving to origin)
    T[0, 3] = -centroid[0]
    T[1, 3] = -centroid[1]
    T[2, 3] = -centroid[2]
    
    # Create a 4x4 identity matrix for scaling
    S = np.eye(4) * scale_factor
    S[3, 3] = 1  # Ensure the last element is 1 to maintain homogeneous coordinates

    # Combine the translation and scaling into one matrix
    # Important: scaling second, translation first in matrix multiplication
    Transformation_matrix = np.dot(S, T)  # This ensures that scaling is applied after translation

    return Transformation_matrix

def create_inverse_transformation_matrix(centroid, scale_factor=1):
    # Inverse scaling matrix
    S_inv = np.eye(4) 
    S_inv[:3, :3] /= scale_factor  # Only apply the scaling to the x, y, z dimensions

    # Inverse translation matrix
    T_inv = np.eye(4)
    T_inv[0, 3] = centroid[0]
    T_inv[1, 3] = centroid[1]
    T_inv[2, 3] = centroid[2]

    # Combine inverse transformations: First apply inverse scaling, then inverse translation
    Transformation_matrix_inv = np.dot(T_inv, S_inv)
    return Transformation_matrix_inv

def check_normalization(mesh_verts, type):
    # Checking max distance from the origin should be <= 1 for both point clouds
    print("check_normalization of {}: {}".format(type, np.max(np.linalg.norm(np.asarray(mesh_verts), axis=1))))
    sys.stdout.flush()
    assert np.max(np.linalg.norm(np.asarray(mesh_verts), axis=1)) <= 1, "{} mesh is not within bounds".format(type)

def check_normalization_random(mesh_verts):
    # Checking max distance from the origin should be <= 1 for both point clouds
    print('check_normalization')
    sys.stdout.flush()
    if np.max(np.linalg.norm(np.asarray(mesh_verts), axis=1)) <= 1:
        return True
    else: return False

def check_unit_cube(mesh_verts, type):
    # Check if the vertices are within the bounds [-1, 1] in all dimensions
    if not np.all((np.asarray(mesh_verts) >= -1) & (np.asarray(mesh_verts) <= 1)):
        # Calculate the excess distances beyond the bounds
        excess_above = np.maximum(np.asarray(mesh_verts) - 1, 0)  # How much is larger than 1
        excess_below = np.maximum(-1 - np.asarray(mesh_verts), 0)  # How much is smaller than -1
        max_excess = np.maximum(np.max(excess_above), np.max(excess_below))
    
        # Raise an assertion error with a message showing the maximum excess distance
        raise AssertionError("{} mesh is not within bounds. Max excess distance: {:.4f}".format(type, max_excess))
    else:
        print("{} mesh is within bounds.".format(type))

#Center and Scale model and data in the same portion
def portion_rescale_GT(pModel_raw, pData_raw, modelMCS_path, dataMCS_path, inv_cubeTform_model=None):
    model_scaled = copy.deepcopy(pModel_raw)
    data_scaled = copy.deepcopy(pData_raw)
    # Process meshes
    centroid, scale_factor = centralize_and_normalize_GT(model_scaled, data_scaled)
    cubeTform = create_transformation_matrix(centroid, scale_factor)
    inv_cubeTform = create_inverse_transformation_matrix(centroid, scale_factor)

    model_scaled.transform(cubeTform)
    data_scaled.transform(cubeTform)
    check_normalization(model_scaled.vertices, 'model')
    check_normalization(data_scaled.vertices, 'data')

    #save mesh
    o3d.io.write_triangle_mesh(modelMCS_path, model_scaled, write_ascii=True)
    o3d.io.write_triangle_mesh(dataMCS_path, data_scaled, write_ascii=True)
    if inv_cubeTform_model is not None:
        np.save(inv_cubeTform_model, inv_cubeTform)

    # return as PointCloud selfect of open3d
    pmodel_scaled = o3d.geometry.PointCloud()
    #pmodel_scaled.points = o3d.utility.Vector3dVector(np.asarray(model_scaled.vertices))
    pmodel_scaled.points = model_scaled.vertices
    
    pdata_scaled = o3d.geometry.PointCloud()
    #pdata_scaled.points = o3d.utility.Vector3dVector(np.asarray(data_scaled.vertices))
    pdata_scaled.points = data_scaled.vertices

    return [pmodel_scaled, model_scaled], [pdata_scaled, data_scaled]

#Center and Scale model and data in the same portion
def portion_rescale(pModel_raw, pData_raw, modelMCS_path, dataMCS_path, inv_cubeTform_model=None, inv_cubeTform_data=None):
    model_scaled = copy.deepcopy(pModel_raw)
    data_scaled = copy.deepcopy(pData_raw)
    # Process meshes
    centroid1, centroid2, scale_factor = centralize_and_normalize(model_scaled, data_scaled)
    cubeTform1 = create_transformation_matrix(centroid1, scale_factor)
    cubeTform2 = create_transformation_matrix(centroid2, scale_factor)
    inv_cubeTform1 = create_inverse_transformation_matrix(centroid1, scale_factor)
    inv_cubeTform2 = create_inverse_transformation_matrix(centroid2, scale_factor)

    model_scaled.transform(cubeTform1)
    data_scaled.transform(cubeTform2)
    check_normalization(model_scaled.vertices, 'model')
    check_normalization(data_scaled.vertices, 'data')
    #check_unit_cube(model_scaled.vertices, 'model')
    #check_unit_cube(data_scaled.vertices, 'data')

    #save mesh
    o3d.io.write_triangle_mesh(modelMCS_path, model_scaled, write_ascii=True)
    o3d.io.write_triangle_mesh(dataMCS_path, data_scaled, write_ascii=True)
    if inv_cubeTform_model is not None and inv_cubeTform_data is not None:
        np.save(inv_cubeTform_model, inv_cubeTform1)
        np.save(inv_cubeTform_data, inv_cubeTform2)

    # return as PointCloud selfect of open3d
    pmodel_scaled = o3d.geometry.PointCloud()
    #pmodel_scaled.points = o3d.utility.Vector3dVector(np.asarray(model_scaled.vertices))
    pmodel_scaled.points = model_scaled.vertices
    
    pdata_scaled = o3d.geometry.PointCloud()
    #pdata_scaled.points = o3d.utility.Vector3dVector(np.asarray(data_scaled.vertices))
    pdata_scaled.points = data_scaled.vertices

    return [pmodel_scaled, model_scaled], [pdata_scaled, data_scaled]

#----------------------------------------------------
"""
# Center and Scale model and data in the same portion
def portion_rescale(pModel_raw, pData_raw, modelMCS_path, dataMCS_path, Tform_scale_center_model_path=None):
        
    # Centering - both with model's center of mass

    model_centered = copy.deepcopy(pModel_raw).translate(-pModel_raw.get_center())
    data_centered = copy.deepcopy(pData_raw).translate(-pModel_raw.get_center())

    # Scale
    max_model = np.abs(np.max(model_centered.vertices))
    max_data = np.abs(np.max(data_centered.vertices))
    min_model = np.abs(np.min(model_centered.vertices))
    min_data = np.abs(np.min(data_centered.vertices))

    max_point = np.max([max_model, max_data, min_model, min_data])
    scale = 1/max_point

    model_scaled = copy.deepcopy(model_centered).scale(scale, center=model_centered.get_center())
    data_scaled = copy.deepcopy(data_centered).scale(scale, center=data_centered.get_center())

    # find inverse transform to scale+center without random - model
    if Tform_scale_center_model_path is not None:
        scaleTform = np.asarray(model_scaled.vertices) / np.asarray(pModel_raw.vertices)
        np.save(Tform_scale_center_model_path, scaleTform)


    #save mesh
    o3d.io.write_triangle_mesh(modelMCS_path, model_scaled, write_ascii=True)
    o3d.io.write_triangle_mesh(dataMCS_path, data_scaled, write_ascii=True)

    # return as PointCloud selfect of open3d
    pmodel_scaled = o3d.geometry.PointCloud()
    pmodel_scaled.points = o3d.utility.Vector3dVector(np.asarray(model_scaled.vertices))
    
    pdata_scaled = o3d.geometry.PointCloud()
    pdata_scaled.points = o3d.utility.Vector3dVector(np.asarray(data_scaled.vertices))

    '''
    # get xyz points
    model = np.asarray(pModel_raw.points)
    data = np.asarray(pData_raw.points)
        
    # Centering - both with model's center of mass 
    model_centered = model - np.mean(model, 0)
    data_centered = data - np.mean(model, 0)

    # Scale
    max_model = np.abs(np.max(model_centered))
    max_data = np.abs(np.max(data_centered))
    min_model = np.abs(np.min(model_centered))
    min_data = np.abs(np.min(data_centered))

    max_point = np.max([max_model, max_data, min_model, min_data])
    scale = 1/max_point

    model_scaled = model_centered*scale
    data_scaled = data_centered*scale

    # return as PointCloud selfect of open3d
    pmodel_scaled = o3d.geometry.PointCloud()
    pmodel_scaled.points = o3d.utility.Vector3dVector(model_scaled)
    if np.any(pModel_raw.colors):
        pmodel_scaled.colors = pModel_raw.colors
    
    pdata_scaled = o3d.geometry.PointCloud()
    pdata_scaled.points = o3d.utility.Vector3dVector(data_scaled)
    if np.any(pData_raw.colors):
        pdata_scaled.colors = pData_raw.colors
    '''
   # data_scaled = mesh and pdata_scaled = point cloud
    return [pmodel_scaled, model_scaled], [pdata_scaled, data_scaled]
"""
def invert_rigidTform(rigidTform):
    R = rigidTform[:3,:3]
    T = rigidTform[:3,3]
    new_T = -R.transpose().dot(T)
    inv_tform = np.identity(4)
    inv_tform[:3,:3] = R.transpose()
    inv_tform[:3,3] = new_T
    return inv_tform

def get_random_transformation(pData_points):
    minpData = np.min(pData_points)
    maxpData = np.max(pData_points)

    # random rotation
    r = -np.pi/2 + np.pi * np.random.rand(1,3)
    rNorm = np.linalg.norm(r)
    rMat = np.array([[0, -r[:,2], r[:,1]] , [r[:,2], 0, -r[:,0]], [-r[:,1], r[:,0], 0]],dtype=object)
    R = np.eye(3) + rMat*(np.sin(rNorm)/rNorm) + rMat.dot(rMat)*((1-np.cos(rNorm))/rNorm**2)

    # random translation
    min = minpData/2
    max = maxpData/2
    T = min + (max-min) * np.random.rand(1,3)

    randTform = np.eye(4)
    randTform[:3, :3] = R
    randTform[:3, 3] = T

    inv_randTform = invert_rigidTform(randTform)

    return randTform, inv_randTform

class DistanceTransform:
    def __init__(self, pcdModel,in_size,in_expandFactor):
        pMin = np.min(np.asarray(pcdModel.points),0)
        pMax = np.max(np.asarray(pcdModel.points),0)

        pCenter = (pMax + pMin) / 2
        pMin = pCenter - in_expandFactor*(pMax-pCenter)
        pMax = pCenter + in_expandFactor*(pMax-pCenter)

        maxTmp = np.max(pMax-pMin)
        pMin = pCenter - maxTmp/2
        pMax = pCenter + maxTmp/2

        scale = in_size/maxTmp
        
        #for now in_size is one number (to be 3 numbers I need to fix scale..)
        #assert len(in_size)==1, "check the specified size of the DT matrix"
        A = np.zeros(np.array([in_size,in_size,in_size]))
        
    
        valuesModel = np.round((np.asarray(pcdModel.points)-pMin)*scale)-1 # -1 for indices in python which start from 0
        idx = np.ravel_multi_index(valuesModel.astype(int).transpose() , (in_size,in_size,in_size))
        #A[idx] = 1
        np.put(A, idx, 1)

        #DT = bwdist(A)/scale
        DT =  ndimage.morphology.distance_transform_edt(1-A)/scale
        
        self.pMin = pMin
        self.pMax = pMax
        self.DT = DT
        self.scale = scale
        self.dt_size = in_size
        self.expandFactor = in_expandFactor
        
        return
    
    def distance(self, pData):
        
        valuesData = np.round((np.asarray(pData.points)-self.pMin)*self.scale)
        
        # handling case of out of range indices values
        # zero/set to dt size the values
        valuesData_new = valuesData;            
        valuesData_new[valuesData_new<1] = 1
        valuesData_new[valuesData_new>self.dt_size] = self.dt_size
        valuesData_new = valuesData_new - 1
        
        # saving the values of the out of range
        outOfArray = np.zeros(valuesData.shape)
        outOfArray[valuesData<1] = valuesData[valuesData<1]
        outOfArray[valuesData>self.dt_size] = valuesData[valuesData> self.dt_size]-self.dt_size
        
        # calculate distance
        idx = np.ravel_multi_index(valuesData_new.astype(int).transpose() , (self.DT.shape))
        minDis = self.DT.ravel()[idx]

        # add the out of range distances
        additionalDistance = np.linalg.norm(outOfArray, axis=1)
        
        # total distance
        minDis = minDis + additionalDistance
        sseDis = np.sum(minDis**2)

        return [sseDis, minDis] 
    
class RotationNode:
    def __init__(self):
        self.a = -np.pi
        self.b = -np.pi
        self.c = -np.pi
        self.w = 2*np.pi
        self.ub = 0
        self.lb = 0
        self.l = 0 
        return

    def __lt__(self ,rotationNode2):
        if self.lb != rotationNode2.lb:
            boolValue = self.lb < rotationNode2.lb
        else:
            boolValue = self.w > rotationNode2.w
        return boolValue
 
    def __gt__(self,rotationNode2):
        if self.lb != rotationNode2.lb:
            boolValue = self.lb > rotationNode2.lb
        else:
            boolValue = self.w < rotationNode2.w
        return boolValue

class TranslationNode:
    def __init__(self):
        self.x = -0.5
        self.y = -0.5
        self.z = -0.5
        self.w = 1.0
        self.ub = 0
        self.lb = 0
        return
    
    def __lt__(self, transNode2):
        if self.lb != transNode2.lb:
            boolValue = self.lb < transNode2.lb
        else:
            boolValue = self.w > transNode2.w
        return boolValue
    
    def __gt__(self, transNode2):
        if self.lb != transNode2.lb:
            boolValue = self.lb > transNode2.lb
        else:
            boolValue = self.w < transNode2.w
        return boolValue

def innerBnB(pData, maxRotDisL, optError, distanceModel, config, initial):
# Input:
# pData - point cloud of Data with the last transformaion
# maxRotDisL - Rotation Uncertainty Radius
# optError - optimal error from outer BNB
# initNodeTrans - default node of transformation
# SSEThresh - threshold
# distanceModel - the model we use, kd tree or DT
# doTrim - true for using trim, else false
# inlierNum
# Output:
# optErrorT - optimal error of inner translation
# nodeTransOut - translation node out

    # initializaition of nodes and queue 
    queueTrans = q.PriorityQueue()
    nodeTrans = TranslationNode()
    nodeTransOut = TranslationNode()

    # taking the outer optimal error
    optErrorT = optError
    queueTrans.put(copy.deepcopy(initial['initNodeTrans']))
    
    # init tanslation transform to default
    translationTform = np.identity(4)

    while True:

        # breaking if the queue if empty
        if queueTrans.empty():
            #nodeTransOut = nodeTrans;                       
            break
        # Push top-level Translation node into priority queue    
        nodeTransParent = queueTrans.get()
        
        # Exit if the optError is less than or equal to the 
        # lower bound plus a small epsilon
        if optErrorT-nodeTransParent.lb < initial['SSEThresh']:
            #nodeTransOut = nodeTrans;                       
            break
        
        # don't go to subcubes if we got to the 4th subcube already
        if np.log2(2/(nodeTransParent.w))==4:
            continue
        
        nodeTrans.w = (nodeTransParent.w)/2
        maxTransDis = (np.sqrt(3)/2.0) * nodeTrans.w

        # Subdivide rotation cube into octant subcubes and 
        # calculate upper and lower bounds for each
        # For each subcube
        for subcube_ii in np.arange(8):

            nodeTrans.x = nodeTransParent.x + (subcube_ii & 1) * nodeTrans.w
            nodeTrans.y = nodeTransParent.y + ((subcube_ii >> 1) & 1) * nodeTrans.w
            nodeTrans.z = nodeTransParent.z + ((subcube_ii >> 2) & 1) * nodeTrans.w

            transX = nodeTrans.x + (nodeTrans.w)/2
            transY = nodeTrans.y + (nodeTrans.w)/2
            transZ = nodeTrans.z + (nodeTrans.w)/2
            
            translationTform[:3, 3] = transX, transY, transZ

            # For each data point, calculate the distance to its 
            # closest point in the model cloud
            # find distances using the kdtree
            pData_copy = copy.deepcopy(pData)
            pDataTemp = pData_copy.transform(translationTform)
            
            if config['is_distance_dt']:
                # use DT metric
                _, minDis = distanceModel.distance(pDataTemp)
                #[error, minDis] = distanceModel.distance(pDataTemp);    
            else:
                # use kdtree metric
                #[~, minDis] = knnsearch(distanceModel, pDataTemp.Location);
                #error = sum(minDis.^2);  -> we dont use the error
                print('KdTree')

            # Subtract the rotation uncertainty radius if calculating the rotation lower bound
            # maxRotDisL == 0 when calculating the rotation upper bound

            #if maxRotDisL != 0:
            if np.any(maxRotDisL):
                minDis = minDis - maxRotDisL;            

            minDis[minDis<0] = 0
            
            if initial['doTrim']:
                #Sort the given array in ascending order
                #Stop immediately after the array is splitted into k small numbers and n-k large numbers
                k_idx = np.argpartition(minDis, initial['inlierNum'])
                minDis = minDis[k_idx[:initial['inlierNum']]]

            # For each data point, find the incremental upper and lower bounds
            ub = np.sum(minDis**2)
            # Subtract the translation uncertainty radius
            dis = minDis - maxTransDis
            lb = np.sum((dis[dis>0])**2)
           
            # If upper bound is better than best, 
            # update optErrorT and optTransOut (optimal translation node)
            if ub < optErrorT:      
                optErrorT = ub
                # think of use isNodeOut - can save run time
                nodeTransOut = copy.deepcopy(nodeTrans)                  

            # Remove subcube from queue if lb is bigger than optErrorT
            if lb >= optErrorT:
                continue

            nodeTrans.ub = ub
            nodeTrans.lb = lb
            queueTrans.put(copy.deepcopy(nodeTrans))

    return optErrorT, nodeTransOut

def downsample_models(pModel_verts, pData_verts):
    if pModel_verts.shape[0] > 5e5:
        pModel_show = pModel_verts[np.sort(np.random.choice(pModel_verts.shape[0], int(0.1*pModel_verts.shape[0]), replace=False)), :]
    else:
        pModel_show = pModel_verts

        
    if pData_verts.shape[0] > 5e5:
        pData_show = pData_verts[np.sort(np.random.choice(pData_verts.shape[0], int(0.5*pData_verts.shape[0]), replace=False)), :]
    else:
        pData_show = pData_verts
    return pModel_show, pData_show

#--------------------------------------------------
# from http://www.open3d.org/docs/release/tutorial/pipelines/icp_registration.html
# and http://www.open3d.org/docs/release/python_api/open3d.pipelines.registration.registration_icp.html

def ICP(source, target, max_dist_th, trans_init=np.identity(4), fitness=0.000001, icpMaxIterations=30, icpThresh= 0.000001):
    # init default is the Identity transform (4x4 matrix)  - np.identity(4)
    
    # the estimation_method isPoint to Point ICP, assumes that the models are in the same scale (with_scale = False)

    # relative_fitness - measures the overlapping area (# of inlier correspondences / # of points in target)
    
    # max_correspondence_distance -  Maximum correspondence points-pair distance
    # This is the radius of distance from each point in the source point-cloud in which the neighbour search will try to find a corresponding point in the target point-cloud.
    # This parameter is most important for performance tuning, as a higher radius will take larger time (as the neighbour search will be performed over a larger radius).

    # source - pData
    # target - pModel

    reg_p2p = o3d.pipelines.registration.registration_icp(
    #reg_p2p = o3d.t.pipelines.registration.icp(
        source=source,
        target=target, 
        max_correspondence_distance =max_dist_th, 
        init=trans_init,
        estimation_method = o3d.pipelines.registration.TransformationEstimationPointToPoint(with_scaling=False),
        #estimation_method = o3d.t.pipelines.registration.TransformationEstimationPointToPoint(with_scaling=False),
        criteria = o3d.pipelines.registration.ICPConvergenceCriteria(relative_fitness=fitness, max_iteration=icpMaxIterations,relative_rmse= icpThresh))#,
        #criteria = o3d.t.pipelines.registration.ICPConvergenceCriteria(relative_fitness=fitness, max_iteration=icpMaxIterations,relative_rmse= icpThresh)#,
        #save_loss_log = True)
    print("Inlier Fitness: ", reg_p2p.fitness)
    print("Inlier RMSE: ", reg_p2p.inlier_rmse)
    print("Correspondence_set size of: ", len(reg_p2p.correspondence_set))
    #print(f'Number of iterations ICP: {(reg_p2p.loss_log["index"].numpy()[-1]) + 1}')
    return reg_p2p

def draw_registration_result(source, target, transformation=np.identity(4)):
    source_temp = copy.deepcopy(source)
    target_temp = copy.deepcopy(target)
    source_temp.paint_uniform_color([0, 1, 0])
    target_temp.paint_uniform_color([1, 0, 1])
    source_temp.transform(transformation)
    if np.asarray(target_temp.points).shape[0] > 5e5:
        target_temp = target_temp.random_down_sample(0.08)
    o3d.visualization.draw_geometries([source_temp, target_temp])
    #zoom=0.4459, front=[0.9288, -0.2951, -0.2242], lookat=[1.6784, 2.0612, 1.4451], up=[-0.3402, -0.9189, -0.1996])
    return

def evaluate_registration(source, target, threshold, trans):
    print("Initial alignment")
    evaluation = o3d.pipelines.registration.evaluate_registration(
        source, target, threshold, trans)
    print(evaluation)
    return evaluation

#--------------------------------------------------
#polyscope

def poly_vis(verts, vertex_scalar=0, faces=0, face_vectors=0, isMesh=False):
    
    # Initialize polyscope
    ps.init()

    if not isMesh: #point cloud
        ### Register a point cloud
        # `verts` is a Nx3 numpy array
        if verts.shape[0] > 5e5:
            verts = verts[np.sort(np.random.choice(verts.shape[0], int(0.1*verts.shape[0]), replace=False)), :]
        ps.register_point_cloud("my points", verts)
    else: #Mesh
        ### Register a mesh
        # `verts` is a Nx3 numpy array of vertex positions
        # `faces` is a Fx3 array of indices, or a nested list
        ps.register_surface_mesh("my mesh", verts, faces, smooth_shade=True)

        # Add a scalar function and a vector function defined on the mesh
        # vertex_scalar is a length V numpy array of values
        # face_vectors is an Fx3 array of vectors per face
        ps.get_surface_mesh("my mesh").add_scalar_quantity("my_scalar", 
                vertex_scalar, defined_on='vertices', cmap='blues')
        ps.get_surface_mesh("my mesh").add_vector_quantity("my_vector", 
                face_vectors, defined_on='faces', color=(0.2, 0.5, 0.5))

    # View the point cloud and mesh we just registered in the 3D UI
    ps.show()

def poly_compare(verts1, verts2):
    # Initialize polyscope
    ps.init()
    verts1, verts2 = downsample_models(verts1, verts2)
    ### Register a point cloud
    # `verts` is a Nx3 numpy array
    ps.register_point_cloud("my points", verts1, color=(1,0,1)) #pink - target (model)
    ps.show() 
    ps.register_point_cloud("my points", verts2, color=(0,1,0)) #green - source (transformed data)
    # View the point cloud and mesh we just registered in the 3D UI
    ps.show()

#--------------------------------------------------
#pyVista

def create_pyVista_obj(vertices, faces=None, colors=None):
    if faces is not None:
        stack_vec = faces.shape[1] * np.ones((faces.shape[0], 1), dtype=int) #number of edges in every face
        pv_object = pv.PolyData(vertices, np.hstack((stack_vec, faces)))

    else:
        pv_object = pv.PolyData(vertices)

    if colors is not None:
        pv_object['colors'] = colors / 256

    return pv_object

def pv_compare(verts1, verts2, title):
    plotter = pv.Plotter()
    #plotter = pv.Plotter(shape=(1, 2))
    plotter.set_background('w')
    plotter.add_title(title, color = 'black')

    pv_mesh1 = create_pyVista_obj(verts1)
    pv_mesh2 = create_pyVista_obj(verts2)
    plotter.add_mesh(pv_mesh1, show_edges=False, color=[0.0,1.0,0.0], preference='point') # green
    plotter.add_mesh(pv_mesh2, show_edges=False, color=[1.0,0.0,1.0], preference='point') # pink

    plotter.link_views()
    plotter.show()
    return plotter

def pv_compare_original_size(verts1, verts2, title):
    
    center1 = verts1.mean(axis=0)
    center2 = verts2.mean(axis=0)

    plotter = pv.Plotter()
    #plotter = pv.Plotter(shape=(1, 2))
    plotter.set_background('w')
    plotter.add_title(title, color = 'black')

    pv_mesh1 = create_pyVista_obj(verts1)
    pv_mesh2 = create_pyVista_obj(verts2)
    plotter.add_mesh(pv_mesh1, show_edges=False, color=[0.0,1.0,0.0], preference='point') # green
    plotter.add_mesh(pv_mesh2, show_edges=False, color=[1.0,0.0,1.0], preference='point') # pink

    # Calculate the center point between two mesh centers
    center_point = (center1 +center2) / 2

    # Set this as the focal point for all linked views
    plotter.camera.focal_point = center_point

    plotter.link_views()
    plotter.show()
    return plotter

def pv_subplot(verts_list):
    #plotter = pv.Plotter()
    n = len(verts_list)
    plotter = pv.Plotter(shape=(1, n))
    plotter.set_background('w')

    for i, verts in enumerate(verts_list):
        pv_mesh1 = create_pyVista_obj(verts[0]) #mesh 1
        pv_mesh2 = create_pyVista_obj(verts[1]) #mesh 2
        plotter.subplot(0, i)
        plotter.add_title(verts[2], color = 'black') #title string
        plotter.add_mesh(mesh=pv_mesh1, color=[0.0,1.0,0.0], preference='point') # green
        plotter.add_mesh(mesh=pv_mesh2, show_edges=False, color=[1.0,0.0,1.0], preference='point') # pink
    plotter.link_views()
    plotter.show()
    return plotter

def pv_show_process(pData, pModel, optArrays):
    verts_list = []
    for i in np.arange(len(optArrays['transformsArray'])):
        tform = optArrays['transformsArray'][i]
        pData_copy = copy.deepcopy(pData)
        pData_copy.transform(tform)
        title = optArrays['typeTransformsArray'][i] + '\n' + 'RMS Error: ' + str(np.round(optArrays['rmsErrorTransformsArray'][i],decimals=3))
        verts_list.append([np.array(pData_copy.points), np.array(pModel.points), title])
    pv_subplot(verts_list)

def pv_visualization_GT(pData, pModel, tform_GT, optArrays, GT_errors):
    verts_list = []
    
    # GT
    pData_copy1 = copy.deepcopy(pData)
    pData_copy1.transform(tform_GT)
    title1 = 'Ground Truth'
    verts_list.append([np.array(pData_copy1.points), np.array(pModel.points), title1])

    # Registration
    pData_copy2 = copy.deepcopy(pData)
    pData_copy2.transform(optArrays['transformsArray'][-1])
    title2 = 'GoICP Registration\n Rotation Error: '+ str(np.round(GT_errors[1],decimals=3)) + " deg\n Translation Error: " + str(np.round(GT_errors[0],decimals=3))
    verts_list.append([np.array(pData_copy2.points), np.array(pModel.points), title2])

    pv_subplot(verts_list)

# for go-icp try
def pv_visualization_GT2(pDataOrig, pDataGoIcp_ans, pModel, GT_errors):
    verts_list = []
    
    # GT
    pData_copy1 = copy.deepcopy(pDataOrig) 
    title1 = 'Ground Truth'
    verts_list.append([np.array(pData_copy1.vertices), np.array(pModel.vertices), title1])

    # Registration
    pData_copy2 = copy.deepcopy(pDataGoIcp_ans)
    title2 = 'GoICP Registration\n Rotation Error: '+ str(np.round(GT_errors[1],decimals=3)) + " deg\n Translation Error: " + str(np.round(GT_errors[0],decimals=3))
    verts_list.append([np.array(pData_copy2.vertices), np.array(pModel.vertices), title2])

    pv_subplot(verts_list)
#--------------------------------------------------
