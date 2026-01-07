import os
import pathlib

def Get_Configuration(model_num, exp_path = '', my_model_type=None, my_type=None):
   #------------------------------------ 
    if my_model_type is None:
        #change as you like 
        model_type = 'C1' #'C1'
        if model_num == 1: #first mesh
            type = '2019' #'2019'
        else: #2
            type = 'goIcp' #second mesh
    else:
        model_type = my_model_type
        if model_num == 1: #first mesh
            type = my_type
        else: #2
            type = 'goIcp' #second mesh
    num_sampels = '300'
    #------------------------------------
    config = {}
    config['model_type'] = model_type
    config['type'] = type
    config['num_sampels'] = num_sampels

    curr_path = pathlib.Path(__file__).parent.resolve() #directory of the script being run
    fast_path = os.path.join(curr_path.parent.absolute(), 'FastSpectrum') 
    #------------------------------------
    
    config['exe_path'] = os.path.join(fast_path,'bin','FastSpectrum.exe')    

    config['meshFile'] = os.path.join(exp_path, config['model_type'] +  config['type'] +'.obj')
    config['saveDest'] = os.path.join(exp_path)
    config['log_path'] = os.path.join(exp_path, 'logfile_fast_')

    return config