import os
import pathlib

#models['model1_path'] - model before changes
#models['model2_path'] - model after changes


# only for experiments, without demo version
# exp_path - experiment path
def Get_Configuration(exp_path = '', my_model1=None, my_model2=None, my_model_type=None, is_GT = None, is_randomTform = None, NdDownsample_num=None, is_vis='None'):
#----------------------------------------------------------
    if my_model1 is None:
        #change to your own
        model1 = '2019' #'2020'
        model2 = '2020' #'2022'

        # for Matan Coral reef data
        model_type = 'C1' #'C2'
    else:
        model1 = my_model1
        model2 = my_model2
        model_type = my_model_type
    
    file_type = 'ply'
#----------------------------------------------------------
    models = {}
    config = {}
    models['model_type'] = model_type
    models['file_type'] = file_type

    #manifold
    file1_name =  models['model_type'] + model1  #for manifold: toManifold = False and permutation, randomTransform,doScale, isRandom=True
    file2_name =  models['model_type'] + model2+ '_R'
        
    # fix models to be manifold
    config['toManifold'] = False #True
    if config['toManifold']:
            directory = 'not_manifold'
    else:
        directory = 'manifold' # I use manifold data

#------------------------------------------------------------------------------------------
    # paths:
    curr_path = pathlib.Path(__file__).parent.resolve() #directory of the script being run
    data_path = os.path.join(curr_path.parent.absolute(), 'data')
    goIcp_path = os.path.join(curr_path.parent.absolute(), 'goIcp')
    ###############################
    
    #raw models
    models['model1_path'] = os.path.join(data_path, directory,  models['model_type'], file1_name + '.' + models['file_type']) #model before changes
    models['model2_path'] = os.path.join(data_path, directory,  models['model_type'], file2_name + '.' + models['file_type']) #model after changes
    assert (os.path.exists(models['model1_path']) or os.path.exists(models['model1_path'])), "One of the file path do not exist."
    
    #save tforms
    models['save_tform_filename'] = os.path.join(exp_path,  models['model_type'] + '_optTform.npy')
    models['save_all_tform'] = os.path.join(exp_path,  models['model_type'] + '_allTform.npy')
    models['save_rmsError_tform_arr'] = os.path.join(exp_path,  models['model_type'] + 'rmsError_tform_arr.npy')
    models['save_typeTransformsArray'] = os.path.join(exp_path,  models['model_type'] + 'typeTransformsArray.npy')

    #permutations save
    models['permutation_filename1'] = os.path.join(exp_path, file1_name + '_permutation.npy')
    models['permutation_filename2'] = os.path.join(exp_path, file2_name + '_permutation.npy')
    
    #GT path
    models['tformGT_path'] = os.path.join(exp_path,  models['model_type'] + '_GTtform.npy')
    
    
    models['random_transformation'] = os.path.join(exp_path,  models['model_type'] + '_randomTform.npy')
    models['inv_random_transformation'] = os.path.join(exp_path,  models['model_type'] + '_invRandomTform.npy')
    # R -  random transform to 2020 if needed
    models['dataR_path'] = os.path.join(exp_path,  models['model_type'] + model2 + '_R.ply') 
    # MCS - manifold, cntered, scaled , random transform to 2020
    models['dataMCSR_path'] = os.path.join(exp_path,  models['model_type'] + model2 + '_MCSR.ply') 

    if config['toManifold']:
        #if not os.path.isdir(os.path.join(goIcp_path, 'results')):
            #os.mkdir(os.path.join(goIcp_path, 'results'))
    
        models['manifold_model1_path'] = os.path.join(data_path, 'manifold',  models['model_type'], file1_name + '__2manifold.' + models['file_type']) #model before changes
        models['manifold_model2_path'] = os.path.join(data_path, 'manifold',  models['model_type'], file2_name + '__2manifold.' + models['file_type']) #model after changes
        models['manifold_log_path1'] = os.path.join(goIcp_path, 'results', 'log_2manifold_model1.txt')
        models['manifold_log_path2'] = os.path.join(goIcp_path, 'results', 'log_2manifold_model2.txt')

    #scale and center
    # MCS - manifold, cntered, scaled
    models['modelMCS'] = os.path.join(exp_path,  models['model_type'] + model1 + '_MCS.ply') 
    models['modelorig'] = os.path.join(exp_path,  models['model_type'] + model1 + '.ply') 
    models['modelorig_obj'] = os.path.join(exp_path,  models['model_type'] + model1 +'.obj') 
    models['dataMCS'] = os.path.join(exp_path,  models['model_type'] + model2 +'_MCS.ply') 

    models['inv_cubeTform_model'] = os.path.join(exp_path,  models['model_type'] + model1 +'inv_TformCS.npy') 
    models['inv_cubeTform_data'] = os.path.join(exp_path,  models['model_type'] + model2 +'inv_TformCS.npy') 

    # path to result of go-icp
    models['goIcp_MCS_path'] = os.path.join(exp_path,  models['model_type'] + 'goIcp_MCS.ply') 
    models['goIcpRes_path'] = os.path.join(exp_path,  models['model_type'] + 'goIcp.ply') 
    models['goIcpRes_obj_path'] = os.path.join(exp_path,  models['model_type'] + 'goIcp.obj') 

    #final Tform - first go-icp then revert the scale center
    models['finalTform'] = os.path.join(exp_path,  models['model_type'] + 'finalTform.npy') 

#------------------------------------------------------------------------------------------
    # config parameters
    
    # triming value
    #if trim_fraction>0.0 config['doTrim'] = true, else false (in initialize)
    if  models['model_type'] == 'C3' or  models['model_type'] == 'C4':
        config['trim_fraction'] = 0 #0.2 
    else: #C1,C2,C5,Kza5m
        #config['trim_fraction'] = 0.1 
        config['trim_fraction'] = 0    

    # flag if i have GT to compare
    models['GTflag'] = True
    if is_GT is not None:
         models['GTflag'] = is_GT

    #Save obj go-icp meshe
    config['isSaveobj'] = True

    # true for permuting the models points
    config['doPermutation'] = True #False
    
    # If the 2 models are aligned (the data is GT) create random transformation to test the algorithm
    config['getRandomTransform'] = True #False
    if is_randomTform is not None:
         config['getRandomTransform'] = is_randomTform
    #if config['getRandomTransform']:
    #     models['tformGT_path'] = models['inv_random_transformation']
    
    # true for rescale data between [-1,1]^3, else false
    config['doScale'] = True #False

    # number of downsampled data (target) points to use in the algorithm. Insert
    # 0 to use the full point cloud, None to use int(np.round(Nm/150/500))*500 
    config['NdDownsampled'] = None
    if NdDownsample_num is not None:
         config['NdDownsampled'] = NdDownsample_num
    #config['NdDownsampled'] = 75000 #150000 
    # isRandom is true for sampling random points and false for taking 
    # NdDownsampled first points
    config['isRandom'] = True #False

    # true for saving the GO-ICP transform
    config['isSaveTform'] = True #False
    
    # rotation subcubes maximum level
    config['maxRotLevel'] = 30 #30
    
    # maximum ICP iterations
    config['icpMaxIterations'] = 400 #50
    
    # maximum loops of outer BnB
    #config['maxLoops'] = 400
    config['maxLoops'] = 1000 #800
    
    # true for using distance DT and false to KD tree -> for now I dont use kdtree
    config['is_distance_dt'] = True
    
    # Mean Squared Error (MSE) convergence threshold
    config['MSEThresh'] = 0.001 # tried 0.0001
    config['icpThresh'] = config['MSEThresh']/10000
    config['max_dist_th_bad'] = 0.5 #run ICP when the prev err is big - find good transformation
    config['max_dist_th_good'] =0.07 #run ICP when the prev err is small - fine tuninng

    config['distTransSize'] = 300
    config['distTransExpandFactor'] = 2.0

    # show visualization
    if is_vis is not None:
         config['isVis'] = is_vis
    else:
        config['isVis'] = False

    return models, config