import matplotlib
#matplotlib.use('TKAgg')
import numpy as np
import open3d as o3d
from goIcp import goIcp_utils #when i run from ex_main
#import goIcp_utils   #when i run from goicp main 
#from sklearn.neighbors import KDTree
import queue as q
import time
import copy
import math
import sys

def write_obj_with_precision(filename, mesh, precision=10):
    with open(filename, "w") as file:
        # Write vertices with specified precision
        for v in mesh.vertices:
            file.write(f"v {v[0]:.{precision}f} {v[1]:.{precision}f} {v[2]:.{precision}f}\n")
        # Write faces (OBJ is 1-indexed)
        for t in mesh.triangles:
            file.write(f"f {t[0]+1} {t[1]+1} {t[2]+1}\n")

def prepare_data(models, config):
    #read the models
    #pModel is the points of the target point set - Fixed
    #pData is the points of the source point set to be transformed - Moved

    #pModel_raw is a list: pModel_raw[0] is point cloud and [1] is triangle mesh, same for pdata
    pModel_raw = goIcp_utils.read_file(models['model1_path'], models['file_type'])
    #o3d.io.write_triangle_mesh(filename=models['modelorig_obj'], mesh=pModel_raw[1],  write_ascii=True, write_vertex_normals=False, 
     #                          write_vertex_colors=False, write_triangle_uvs=False) #save as obj
    write_obj_with_precision(models['modelorig_obj'], pModel_raw[1])
    o3d.io.write_triangle_mesh(filename=models['modelorig'], mesh=pModel_raw[1],  write_ascii=True)
    pData_raw = goIcp_utils.read_file(models['model2_path'], models['file_type'])
    Nm = len(np.asarray(pModel_raw[0].points))
    Nd = len(np.asarray(pData_raw[0].points))

    #------------------------------------------
    # scale and normalize the models into [-1,1]^3
    if config['doScale']:
        print("Scale and center")
        if models['GTflag']:
            # pModel_tmp is  a list: [0] is point cloud and [1] is mesh
            pModel_tmp, pData_tmp = goIcp_utils.portion_rescale_GT(pModel_raw[1], pData_raw[1], models['modelMCS'], models['dataMCS'],
                                                             models['inv_cubeTform_model'])
        else:
            # pModel_tmp is  a list: [0] is point cloud and [1] is mesh
            pModel_tmp, pData_tmp = goIcp_utils.portion_rescale(pModel_raw[1], pData_raw[1], models['modelMCS'], models['dataMCS'],
                                                                models['inv_cubeTform_model'],models['inv_cubeTform_data'])
        pModel = pModel_tmp[0]
        pData = pData_tmp[0]
        #pModelMesh = pModel_tmp[1]
        pDataMesh =  pData_tmp[1]
    
    else:
        pModel =  pModel_raw[0]
        pData = pData_raw[0]
        #pModelMesh = pModel_raw[1]
        pDataMesh =  pData_raw[1]

    #utils.draw_registration_result(pData_raw, pModel_raw)
    #utils.draw_registration_result(pData, pModel)

    #------------------------------------------
    # permute the models points
    if config['doPermutation']:
        print("Do permutation")
        p1 = np.random.permutation(Nm) #for pModel
        pModel_perm = o3d.geometry.PointCloud()
        pModel_perm.points = o3d.utility.Vector3dVector(np.array(pModel.points)[p1,:])
        p2 = np.random.permutation(Nd) #for pData
        pData_perm = o3d.geometry.PointCloud()
        pData_perm.points = o3d.utility.Vector3dVector(np.array(pData.points)[p2,:])
        
        # save permutations for post processing GO-ICP
        print("Save permutations")
        np.save(models['permutation_filename1'], p1)
        np.save(models['permutation_filename2'], p2)

        pModelAlg =  pModel_perm
        pData = pData_perm

    else:
        pModelAlg =  pModel
    #utils.draw_registration_result(pData, pModelAlg)
    #------------------------------------------
    # Random transformation
    # If the 2 models are aligned (the data is GT) create random transformation to test the algorithm
    if config['getRandomTransform']:
        print("Calc Random Transform")
        # get random transformation 
        randTform, inv_randTform = goIcp_utils.get_random_transformation(np.array(pData.points))
        pData.transform(randTform)

        if not goIcp_utils.check_normalization_random(pData.points):
            print("Fix Random Transform to a cube")
            centroid = np.mean(np.array(pData.points), axis=0)
            Transformation_matrix = goIcp_utils.create_transformation_matrix(centroid)
            pData.transform(Transformation_matrix)
            Transformation_matrix_inv = goIcp_utils.create_inverse_transformation_matrix(centroid)
            goIcp_utils.check_normalization(pData.points, 'data after random')
            randTform = np.dot(Transformation_matrix, randTform)
            inv_randTform = np.dot(inv_randTform, Transformation_matrix_inv)

        #save random transformation
        np.save(models['random_transformation'], randTform)
        np.save(models['inv_random_transformation'], inv_randTform)

        # save data with random tform as mesh - no permutation
        pDataRT =copy.deepcopy(pDataMesh).transform(randTform) #pData_tmp[0] is a mesh after center and scale - with no permutation
        o3d.io.write_triangle_mesh(models['dataMCSR_path'], pDataRT, write_ascii=True)

        

    #utils.draw_registration_result(pData, pModelAlg)
    #------------------------------------------

    # Downsample the target points
    print(f'Fixed model includes {Nm} samples')

    # if NdDownsampled is not specified
    if config['NdDownsampled'] is None:
        # in case of Nm large:
        config['NdDownsampled'] = int(np.round(Nm/150/500))*500
        
        if config['NdDownsampled'] < 1000:
            config['NdDownsampled'] = 1000
        
        
    if config['NdDownsampled'] > 0:
        
        NdDownsampled = config['NdDownsampled']
        print(f'Moving model is downsampled from {Nd} to {NdDownsampled} samples')
        
        percentage = config['NdDownsampled']/Nd
        Nd = config['NdDownsampled']

        if config['isRandom']:
            # sample randomly
            print(f'Random downsample')
            pDataAlg = pData.random_down_sample(percentage)
        else:
            # take Nd first points
            print(f'Downsample: take Nd first points')
            pDataAlg = o3d.geometry.PointCloud()
            points = np.array(pData.points)
            pDataAlg.points = o3d.utility.Vector3dVector(points[:Nd,:])
            if np.any(pData_raw.colors):
                pDataAlg.colors = pData_raw.colors[:Nd,:]
            
    else: #if NdDownsampled=0 , take full models
        pDataAlg = pData
        print(f'Moving model is NOT downsampled It includes {np.array(pDataAlg.points).shape[0]} samples')

    #utils.draw_registration_result(pDataAlg, pModelAlg)
    
    return pModelAlg, pDataAlg, Nd


#initial['DT']
#initial['kdtreeModel'] -> for now dont use kdtree
#initial['doTrim']
#initial['inlierNum']
#initial['icpInlierRatio']
#initial['SSEThresh']
#initial['queueRot']
#initial['lb']
#initial['initNodeTrans']
#initial['maxRotDis']


#optArrays['countArray']
#optArrays['transformsArray']
#optArrays['typeTransformsArray']
#optArrays['ssErrorTransformsArray']
#optArrays['rmsErrorTransformsArray']

#nodes['nodeRot']
#nodes['optNodeRot']
#nodes['optNodeTrans']

#Go ICP initialization
def initialize(pModel, pData, Nd, config):
    initial = {}
    optArrays = {}
    nodes = {}

    # 3D distance metric
    if config['is_distance_dt']:
        #calculate 3D Distance transform array
        print('Distance Calculation by Distance Transform metric\n Building Distance Transform Matrix...')
        initial['DT'] = goIcp_utils.DistanceTransform(pModel, config['distTransSize'], config['distTransExpandFactor'])
    else:
        # calculate kd-tree object
        print('Distance Calculation by kd-Tree metric\n Building kd-Tree object...')
        #kdtreeModel = KDTreeSearcher(pModel.Location);
        #initial['kdtreeModel'] = KDTree(np.asarray(pModel.points))

    #------------------------------------------
    # initializing BnB parameters
    # rotation and translation cubes
    print('Init BnB parameters')
    initNodeRot = goIcp_utils.RotationNode()
    initial['initNodeTrans'] = goIcp_utils.TranslationNode()
    nodes['nodeRot'] = goIcp_utils.RotationNode()

    # array of all the opt transformations in the process
    optArrays['countArray'] = []
    optArrays['transformsArray'] = []
    optArrays['typeTransformsArray'] = []
    optArrays['ssErrorTransformsArray'] = []
    optArrays['rmsErrorTransformsArray'] = []

    # init the rotation priority queue
    #queueRot = PQueue('RotationNode')
    initial['queueRot'] = q.PriorityQueue()
    # init low bound
    initial['lb'] = np.inf
    #------------------------------------------   
    # Compute maximum rotations for each subcube level
    # Precompute the rotation uncertainty distance (maxRotDis) for each point 
    # in the data and each level of rotation subcube

    # Calculate L2 norm of each point in data cloud to origin
    normData = np.linalg.norm(np.asarray(pData.points), axis=1)
    maxRotLevelVec = np.arange(1, config['maxRotLevel']+1)
    # Half-side length of each level of rotation subcube
    sigma = initNodeRot.w / (2**maxRotLevelVec)
    maxAngle = np.sqrt(3) * sigma
    maxAngle[maxAngle>np.pi] = np.pi
    initial['maxRotDis'] = 2 * normData[..., None] * np.sin(maxAngle/2)

    #------------------------------------------
    # Initialise so-far-best rotation and translation nodes(cubes) and matrices
    nodes['optNodeRot'] = initNodeRot
    nodes['optNodeTrans'] = copy.deepcopy(initial['initNodeTrans'])

    # Initialise so-far-best rotation and translation 
    #optR = np.eye(3)
    #optT = np.zeros([1,3])
    #optTform = rigid3d(optR, optT) -> see if I can find rigid3d in python
    optTform = np.identity(4)

    # insert to transformsArray for tracking
    countOptTform = 1
    optArrays['countArray'].append(countOptTform)
    optArrays['transformsArray'].append(optTform)
    optArrays['typeTransformsArray'].append('Initialized')
    #countOptTform = countOptTform + 1

    #------------------------------------------
    # trim
    if config['trim_fraction'] > 0.0:
        initial['doTrim'] = True
    else:
        initial['doTrim'] = False
        
    # For untrimmed ICP, use all points, otherwise only use inlierNum points
    if initial['doTrim']:
        initial['inlierNum'] = int(Nd * (1-config['trim_fraction']))
        print('Trimming, number of inliers: {}'.format(initial['inlierNum']))
        #initial['icpInlierRatio'] = 1-config['trim_fraction']
    else:
        initial['inlierNum'] = int(Nd)
        print('No Trimming')
        #initial['icpInlierRatio'] = 1

    # Sum of Sqauare Error Threshold
    initial['SSEThresh'] = config['MSEThresh'] * initial['inlierNum']
    
    return initial, optArrays, nodes


#Go ICP initialization
def outerBNB(pModel, pData, initial, optArrays, nodes, config, models):
    
    start = time.time()
    optTform = np.identity(4)
    #find initial error
    if config['is_distance_dt']:
        # use DT metric
        _, minDis = initial['DT'].distance(pData)
    else:
        # use kdtree metric
        # find distances using the kdtree
        #_, minDis = knnsearch(initial['kdtreeModel'], np.asarray(pData.points))
        print('KdTree')

    if initial['doTrim']:
        # Sort the given array in ascending order
        # Stop immediately after the array is splitted into k small numbers and n-k large numbers
        #minDis = mink(minDis,initial['inlierNum'])
        k_idx = np.argpartition(minDis, initial['inlierNum'])
        minDis = minDis[k_idx[:initial['inlierNum']]]

    optError = np.sum(minDis**2)
    optArrays['ssErrorTransformsArray'].append(optError) 
    optArrays['rmsErrorTransformsArray'].append(np.sqrt(optError/minDis.shape[0]))
    print(f'Initialized Error*: {optError}')
    countOptTform = 2
    
    #------------------------------------------
    # Run ICP from initial state
    print('Run ICP from initial state')
    #threshold ?= initial['icpInlierRatio'] -> not exactly
    Nm = len(np.asarray(pModel.points))
    #icpfitness = initial['icpInlierRatio'] / Nm
    reg_p2p = goIcp_utils.ICP(source=pData, target=pModel, max_dist_th=config['max_dist_th_bad'] , icpMaxIterations=config['icpMaxIterations'], icpThresh = config['icpThresh'])#, fitness=icpfitness)
    tformICP = copy.deepcopy(reg_p2p.transformation)
    tformICP.flags.writeable = True

    #------------------------------------------
    # compute current error

    if config['is_distance_dt']:
        # use DT metric
        pData_copy = copy.deepcopy(pData)
        pDataTempICP = pData_copy.transform(tformICP)
        _, minDis = initial['DT'].distance(pDataTempICP)
    else:
        # use kdtree metric
        # find distances using the kdtree
        #_, minDis = knnsearch(initial['kdtreeModel'], np.asarray(pData.points))
        print('KdTree')
    
    if initial['doTrim']:
        # Sort the given array in ascending order
        # Stop immediately after the array is splitted into k small numbers and n-k large numbers
        #minDis = mink(minDis,initial['inlierNum'])
        k_idx = np.argpartition(minDis, initial['inlierNum'])
        minDis = minDis[k_idx[:initial['inlierNum']]]

    # compute error
    error = np.sum(minDis**2)
    print(f'Initialized ICP - Error*: {error}')

    # compare with optimal error
    if error < optError:
        optError = error
        optTform = tformICP
        optArrays['countArray'].append(countOptTform)
        optArrays['transformsArray'].append(optTform) 
        optArrays['typeTransformsArray'].append('ICP')
        optArrays['ssErrorTransformsArray'].append(optError) 
        #rmsError = copy.deepcopy(reg_p2p.inlier_rmse)
        optArrays['rmsErrorTransformsArray'].append(np.sqrt(optError/minDis.shape[0]))
        countOptTform +=  1

    # Push top-level rotation node into priority queue
    initial['queueRot'].put(copy.deepcopy(nodes['optNodeRot'])) #initial Node Rotation

    # Counters init
    count = 0
    countICP = 0

    #------------------------------------------
    # Keep exploring rotation space until convergence is achieved
    print('Keep exploring rotation space until convergence is achieved')
    while True:
        
        # break when no more appropiate cubes
        if initial['queueRot'].empty():
            print('Rotation Queue Empty')
            print(f'Number of Loops: {count}')
            lb_print = initial['lb']
            print(f'Error*: {optError}, LB: {lb_print}')
            last = optArrays['rmsErrorTransformsArray'][-1]
            print(f'RMS Error: {last}')
            break
        
        # Access rotation cube with lowest lower bound and remove it from the queue
        nodeRotParent = initial['queueRot'].get()
        
        # Exit if the optError is less than or equal to 
        # the lower bound plus a small epsilon
        if (optError - nodeRotParent.lb <= initial['SSEThresh']):
            SSEThresh_print = initial['SSEThresh']
            print(f'optError <= lower bound + small epsilon')
            print(f'Number of Loops: {count}')
            print(f'Error*: {optError}, LB: {nodeRotParent.lb}, epsilon: {SSEThresh_print}')
            last_rmsError = optArrays['rmsErrorTransformsArray'][-1]
            print(f'RMS Error: {last_rmsError}')
            break

        # print every 200 loops
        if (count>0) and (np.mod(count,200) == 0):
            print(f'Loop #{count}')
            print(f'LB = {nodeRotParent.lb}, L = {nodeRotParent.l} ')   
        
        # Reached to maximum number of loops
        if (count>0) and (np.mod(count,config['maxLoops'] ) == 0):
            maxLoops_print = config['maxLoops']
            print(f'Reached to maximum number of loops: {maxLoops_print}')
            print(f'Number of Loops: {maxLoops_print}')
            SSEThresh_print = initial['SSEThresh']
            print(f'Error*: {optError}, LB: {nodeRotParent.lb}, epsilon: {SSEThresh_print}')
            last_rmsError = optArrays['rmsErrorTransformsArray'][-1]
            print(f'RMS Error: {last_rmsError}')

            break
        
        count += 1
        
        # Subdivide rotation cube into octant subcubes and calculate upper and lower bounds for each
        nodes['nodeRot'].w = (nodeRotParent.w)/2
        nodes['nodeRot'].l = nodeRotParent.l + 1

        # For each subcube
        for subcube_ii in np.arange(8): #0-7
            # Calculate the smallest rotation across each dimension
            nodes['nodeRot'].a = nodeRotParent.a + ((subcube_ii & 1) * nodes['nodeRot'].w)
            nodes['nodeRot'].b = nodeRotParent.b + (((subcube_ii >> 1) & 1) * nodes['nodeRot'].w)
            nodes['nodeRot'].c = nodeRotParent.c + (((subcube_ii >> 2) & 1) * nodes['nodeRot'].w)
        
            # Find the subcube center
            r = np.array([nodes['nodeRot'].a, nodes['nodeRot'].b, nodes['nodeRot'].c]) + nodes['nodeRot'].w/2
            rNorm = np.linalg.norm(r)
            
            # Skip subcube if it is completely outside the rotation PI-ball
            if (rNorm - (np.sqrt(3)*(nodes['nodeRot'].w)/2)) > np.pi:
                continue
            
            # Convert angle-axis rotation into a rotation matrix        
            if rNorm > 0:
                
                rMat = np.array([[0, -r[2], r[1]] , [r[2], 0, -r[0]], [ -r[1], r[0], 0]])
                R = np.eye(3) + rMat*(np.sin(rNorm)/rNorm) + rMat.dot(rMat)*((1-np.cos(rNorm))/rNorm**2)
                
                # Rotation Matrix is tranposed in Matlab - pctransform
                rotation = np.eye(4)
                rotation[:3, :3] = R
                rotation[:3, 3] = 0
                #rotation = rotation.transpose() # NO, not as rigid3d in matlab - in python I dont need to transpose 

                # rotate the moving point cloud
                pData_copy = copy.deepcopy(pData)
                pDataTemp = pData_copy.transform(rotation) # need to be same as matlab 

                #utils.draw_registration_result(pData, pModel)
                #utils.draw_registration_result(pDataTemp, pModel)
            
            #  If rNorm == 0, the rotation angle is 0 and no rotation is required    
            else:
                pDataTemp = pData
            
            # Upper Bound
            # Run Inner Branch-and-Bound to find rotation upper bound
            # Calculates the rotation upper bound by finding the translation 
            # upper bound for a given rotation,
            # assuming that the rotation is known (zero rotation uncertainty radius)
            maxRotDisL = 0
            if config['is_distance_dt']:
                #optError_test = 5.3114
                #ub, nodeTransOut = goIcp_utils.innerBnB(pDataTemp, maxRotDisL, optError_test, initial['DT'], config, initial)
                ub, nodeTransOut = goIcp_utils.innerBnB(pDataTemp, maxRotDisL, optError, initial['DT'], config, initial)
            else:
                #ub, nodeTransOut = goIcp_utils.innerBnB(pDataTemp, maxRotDisL, optError, initial['kdtreeModel'],  config, initial)
                print('KdTree')

            # If the upper bound is the best so far, run ICP
            if ub < optError:
                
                countICP = countICP + 1
                
                # Update optimal error and rotation/translation nodes
                optError = ub
                optNodeRot = copy.deepcopy(nodes['nodeRot'])
                optNodeTrans = copy.deepcopy(nodeTransOut)
                optTform = np.eye(4)           
                optTform[:3, :3] = R
                # go to center of cube from innerBnB
                optTform[:3, 3] = optNodeTrans.x + optNodeTrans.w/2, optNodeTrans.y+ optNodeTrans.w/2, optNodeTrans.z + optNodeTrans.w/2
                
                # assign into the optimal transformations tracking array 
                optArrays['countArray'].append(countOptTform)             
                optArrays['transformsArray'].append(optTform)
                optArrays['typeTransformsArray'].append('BNB')
                optArrays['ssErrorTransformsArray'].append(optError) 
                optArrays['rmsErrorTransformsArray'].append(np.sqrt(optError/minDis.shape[0]))
                countOptTform +=  1

                print(f'Error*: {optError}. **Before** ICP {countICP}')

                # Run ICP from last transformed state
                pData_copy = copy.deepcopy(pData)
                pDataTemp2icp = pData_copy.transform(optTform)
                reg_p2p = goIcp_utils.ICP(source=pDataTemp2icp, target=pModel, max_dist_th=config['max_dist_th_good'], icpMaxIterations=config['icpMaxIterations'], icpThresh = config['icpThresh'])#, fitness=icpfitness)
                tformICP = copy.deepcopy(reg_p2p.transformation)
                tformICP.flags.writeable = True
                
                # compute current error
                if config['is_distance_dt']:
                    # use DT metric
                    pData_copy = copy.deepcopy(pData)
                    #pDataTempICP = pData_copy.transform(optTform * tformICP)
                    pDataTempICP = pData_copy.transform(tformICP.dot(optTform))
                    _, minDis = initial['DT'].distance(pDataTempICP)
                else:
                    # use kdtree metric
                    #pDataTempICP = pctransform(pDataAlg, rigid3d(optTform.T * tformICP.T));
                    #[~, minDis] = knnsearch(kdtreeModel, pDataTempICP.Location);
                    print('KdTree')
                
                if initial['doTrim']:
                    #Sort the given array in ascending order
                    #Stop immediately after the array is splitted into k small numbers and n-k large numbers
                    k_idx = np.argpartition(minDis, initial['inlierNum'])
                    minDis = minDis[k_idx[:initial['inlierNum']]]
                
                # compute error
                error = np.sum(minDis**2)
                print(f'Error*: {error}. **After** ICP {countICP}')
                
                # check icp error
                if error < optError:
                    optError = error
                    # need to add new transform to old transform
                    #optTform = optTform * tformICP 
                    optTform = tformICP.dot(optTform)
                    # assign into the optimal transformations tracking array
                    optArrays['countArray'].append(countOptTform)
                    optArrays['transformsArray'].append(optTform) 
                    optArrays['typeTransformsArray'].append('ICP')
                    optArrays['ssErrorTransformsArray'].append(optError) 
                    optArrays['rmsErrorTransformsArray'].append(np.sqrt(optError/minDis.shape[0]))
                    countOptTform +=  1

                # Discard all rotation nodes with high lower bounds in the queue
                queueRotNew = q.PriorityQueue()
                while not initial['queueRot'].empty():
                    node = initial['queueRot'].get()
                    if node.lb < optError:
                        queueRotNew.put(copy.deepcopy(node))
                    else:
                        break
                
                initial['queueRot'].queue = copy.deepcopy(queueRotNew.queue)
                    
            
            # Lower Bound
            # Run Inner Branch-and-Bound to find rotation lower bound
            # Calculates the rotation lower bound by finding the translation upper bound for a given rotation,
            # assuming that the rotation is uncertain (a positive rotation uncertainty radius)
            # Pass an array of rotation uncertainties for every point in data cloud at this level
            
            maxRotDisL = initial['maxRotDis'][:, nodes['nodeRot'].l]
            if config['is_distance_dt']:
                initial['lb'], _ = goIcp_utils.innerBnB(pDataTemp, maxRotDisL, optError, initial['DT'], config, initial)
            else:
                #lb, _ = goIcp_utils.innerBnB(pDataTemp, maxRotDisL, optError, initial['kdtreeModel'],  config, initial)
                print('KdTree')
            
            # If the best error so far is less than the lower bound, 
            # remove the rotation subcube from the queue
            if initial['lb'] >= optError:
                continue
            
            # Update node and put it in queue
            nodes['nodeRot'].ub = ub
            nodes['nodeRot'].lb = initial['lb']
            initial['queueRot'].put(copy.deepcopy(nodes['nodeRot']))

    end = time.time()
    print(f'Runtime outerBNB {end-start} sec.')
    #------------------------------------------
    # Finished algorithm running

    #save GO-ICP opt transform
    if config['isSaveTform']:
        np.save(models['save_tform_filename'], optTform)
        np.save(models['save_all_tform'], optArrays['transformsArray'])
        np.save(models['save_rmsError_tform_arr'], optArrays['rmsErrorTransformsArray'])
        np.save(models['save_typeTransformsArray'], optArrays['typeTransformsArray'])

    print("Finish Algorithm Running")
    return

def computeTformError(models, optArrays):
    
    tform_GT = np.load(models['inv_random_transformation'],allow_pickle=True) 
    transError = np.linalg.norm(optArrays['transformsArray'][-1][:3, 3] - tform_GT[:3, 3])

    # rotationError is the angular distance between rotations
    rotationError = (optArrays['transformsArray'][-1][:3, :3].transpose()).dot(tform_GT[:3, :3]) #angular distance
    rotationError[np.logical_and(rotationError<0, rotationError>(-1e-6))] = 0 
    # thetaError in degrees - the angle of the error
    #r = np.arccos(0.5*(np.trace(optArrays['transformsArray'][-1][:3, :3])-1))???
    thetaError = math.degrees(np.arccos(0.5*(np.trace(rotationError)-1)))

    print(f'Translation Error: {transError}')
    print(f'Rotation Error: {thetaError} degrees')
    
    return tform_GT, [transError,thetaError]

#  exp_directory - directory to save experimrnts
#def main(config_goIcp, exp_directory=''):
def main(models, config):

    start = time.time()
    #models, config = config_goIcp.Get_Configuration(exp_directory)

    if config['toManifold']: # create manifold data
        assert goIcp_utils.manifold(models,  models['model_type']), "The Models are not manifold"

    print(f'Prepare Data - Preprocessing') 
    # read the models, downsampling
    pModel, pData, Nd = prepare_data(models, config)
    sys.stdout.flush()
    #start = time.time()

    print(f'Go-ICP initialization') 
    # nodes(cubes), trim, maximum rotation distance, SSE threshold, distance metric
    initial, optArrays, nodes = initialize(pModel, pData, Nd, config)
    sys.stdout.flush()

    # Go ICP Outer BnB - the main Algoritm
    print('Start BNB')
    outerBNB(pModel, pData, initial, optArrays, nodes, config, models)
    sys.stdout.flush()

    end = time.time()
    total_time = end-start
    print(f'Runtime Go-Icp {total_time} sec.')

    # save the data after GoIcp
    optTform = optArrays['transformsArray'][-1]
    if models['GTflag']:
        pData_notReg = o3d.io.read_triangle_mesh(filename=str(models['dataMCSR_path'])) #load data after scale center and random tform (no permutation)
    else:
        pData_notReg = o3d.io.read_triangle_mesh(filename=str(models['dataMCS'])) #load data after scale center (no permutation)
        
    pDataGoIcp =copy.deepcopy(pData_notReg).transform(optTform)
    o3d.io.write_triangle_mesh( models['goIcp_MCS_path'], pDataGoIcp, write_ascii=True)

    if config['doScale']:
        inv_cubeTform = np.load(models['inv_cubeTform_model'],allow_pickle=True)
        pDataGoIcp_ans =copy.deepcopy(pDataGoIcp).transform(inv_cubeTform)
        finalTform  = np.dot(inv_cubeTform, optTform)
    else: 
        pDataGoIcp_ans = pDataGoIcp
        finalTform = optTform

    o3d.io.write_triangle_mesh( models['goIcpRes_path'], pDataGoIcp_ans, write_ascii=True) #save final ans after go-icp and revert scale center
    np.save(models['finalTform'], finalTform)
    if config['isSaveobj']:
        #o3d.io.write_triangle_mesh(filename=models['goIcpRes_obj_path'], mesh=pDataGoIcp_ans, write_ascii=True,
        #                        write_vertex_normals=False, write_vertex_colors=False, write_triangle_uvs=False) #save final ans after go-icp and revert scale center as obj
        write_obj_with_precision(models['goIcpRes_obj_path'], pDataGoIcp_ans)

    if config['isVis']:
        # Visualization - Process of the scaled data
        goIcp_utils.pv_show_process(pData, pModel, optArrays)
        
        # Visualization - Comparison
        comp_title1 = 'Registered After Scaling'  +  models['model_type']
        goIcp_utils.pv_compare(np.array(pDataGoIcp.vertices), np.array(pModel.points), comp_title1)
        if config['doScale']:
            modelorig = o3d.io.read_triangle_mesh(filename=str( models['modelorig']))
            comp_title2 = 'Registered Original Size'  +  models['model_type']
            goIcp_utils.pv_compare_original_size(np.array(pDataGoIcp_ans.vertices), np.array(modelorig.vertices), comp_title2)

    # Rotation and translation error, in case GT
    if models['GTflag']:
       tform_GT, GT_errors = computeTformError(models, optArrays)
       if config['isVis']:
           goIcp_utils.pv_visualization_GT(pData, pModel, tform_GT, optArrays, GT_errors)
       sys.stdout.flush()
       return total_time, GT_errors
    
    sys.stdout.flush()
    return total_time 


if __name__ == '__main__':

    import config as config_goIcp # this config is not updated
    models, config = config_goIcp.Get_Configuration()
    main(models, config)
