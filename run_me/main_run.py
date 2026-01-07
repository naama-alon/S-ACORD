import matplotlib
matplotlib.use('TKAgg')
import numpy as np
import pathlib, os
import sys
sys.path.append(os.path.join(pathlib.Path(__file__).parent.resolve().parent.absolute())) # main project path
from goIcp import goicp
from FastSpectrum import FastSpectrum
from functional_map import fmap
from analysis import analysis
import datetime
#import statistics
#import matplotlib.pyplot as plt
import math
import time
import glob

def calculate_total_error(translation_error, rotation_error):
    # If your rotation error is in degrees, you might want to convert it to radians,
    # or ensure both errors are in comparable units
    # rotation_error_radians = math.radians(rotation_error)

    total_error = math.sqrt(translation_error**2 + rotation_error**2)
    return total_error


def main(loops, inputs):

    goIcp_error= [] 
    total_error = []
    goIcp_times =[] 
    FastSpectrum_times =[] 
    fmap_times =[] 
    analysis_times =[] 
    total_times = []

    start = time.time()

    # Get the current date and time
    curr_datetime = datetime.datetime.now().strftime("%Y%m%d_%H_%M")
    main_path = os.path.join(pathlib.Path(__file__).parent.resolve(),'results',curr_datetime)

    if not os.path.exists(main_path):
        os.makedirs(main_path)

    #create README file
    with open(os.path.join(main_path,'README.txt'), 'w') as file:
        file.write("Name of coral: {}, model 1:{}, model 2:{}".format(inputs['model_type'],inputs['model1'],inputs['model2']))

    n_loops = np.arange(loops)
    for i in n_loops:
        
        loop_path = os.path.join(main_path,str(i))
        if not os.path.exists(loop_path):
            os.makedirs(loop_path)
        # Open a file where you want to log the prints
        log_file = open(os.path.join(loop_path, 'logfile_' + str(i) +'.log'), 'w')

        # Redirect print output to the log file
        sys.stdout = log_file
        try:
            print("Name of coral: {}, model 1:{}, model 2:{}".format(inputs['model_type'],inputs['model1'],inputs['model2']))

            #start GOICP
            print('start go-icp')
            if inputs['is_variables']: 
                models_goIcp, config_goIcp = inputs['goIcp'].Get_Configuration(loop_path,  inputs['model1'], inputs['model2'],
                                                                    inputs['model_type'],inputs['is_GT'],inputs['is_randomTform'],inputs['NdDownsample_num'], inputs['is_vis'])
            else: #use default
                models_goIcp, config_goIcp = inputs['goIcp'].Get_Configuration(exp_directory=loop_path)

            if inputs['is_GT']:
                time_goIcp, GT_errors = goicp.main(models_goIcp, config_goIcp)
                sys.stdout.flush()
                GT_total_error = calculate_total_error(GT_errors[0],GT_errors[1])
                goIcp_error.append(GT_errors)
                total_error.append(GT_total_error)
            else:
                time_goIcp = goicp.main(models_goIcp, config_goIcp)
                sys.stdout.flush()
                 
            goIcp_times.append(time_goIcp)
            print("Timing of go-icp: {} sec".format(time_goIcp))
            
            
            print('start FastSpectrum')
            
            if inputs['is_variables']:
                config_fast1 = inputs['FastSpectrum1'][0].Get_Configuration(inputs['FastSpectrum1'][1], loop_path, 
                                                                            inputs['model_type'], inputs['model1'])
                config_fast2 = inputs['FastSpectrum2'][0].Get_Configuration(inputs['FastSpectrum2'][1], loop_path, 
                                                                            inputs['model_type'], inputs['model2'])
            else:
                config_fast1 = inputs['FastSpectrum1'][0].Get_Configuration(inputs['FastSpectrum1'][1],exp_directory=loop_path)
                config_fast2 = inputs['FastSpectrum2'][0].Get_Configuration(inputs['FastSpectrum2'][1],exp_directory=loop_path)

            time_Fast1 = FastSpectrum.main(config_fast1)
            sys.stdout.flush()
            time_Fast2 = FastSpectrum.main(config_fast2)
            sys.stdout.flush()

            FastSpectrum_times.append(time_Fast1+time_Fast2)
            print("Timing of FastSpectrum: {} sec".format(FastSpectrum_times[-1]))


            print('start fmap')
            if inputs['is_variables']:
                config_fmap = inputs['fmap'].Get_Configuration(loop_path, inputs['model_type'], inputs['model1'], inputs['is_vis'])
            else:
                config_fmap = inputs['fmap'].Get_Configuration(exp_directory=loop_path)

            time_fmap = fmap.main(config_fmap)
            sys.stdout.flush()

            fmap_times.append(time_fmap)
            print("Timing of fmp: {} sec".format(time_fmap))

            print('start analysis')
            if inputs['is_variables']:
                config_analysis = inputs['analysis'].Get_Configuration(loop_path, inputs['model_type'], inputs['model1'], inputs['is_vis'])
            else:
                config_analysis = inputs['analysis'].Get_Configuration(exp_directory=loop_path)

            time_analysis = analysis.main(config_analysis)
            sys.stdout.flush()
            
            analysis_times.append(time_analysis)
            print("Timing of analysis: {} sec".format(time_analysis))

            total_times.append(time_goIcp+time_Fast1+time_Fast2+time_fmap+time_analysis)

            print('end algorithm')
            print("Total Timing: {} sec".format(total_times[-1]))
        finally:
            sys.stdout.flush()
            
            # Important: Close the log file when done
            log_file.close()

            # Reset sys.stdout to its default value if needed
            sys.stdout = sys.__stdout__
        
        if inputs['is_delete']:
            # List all files in the directory
            all_files = glob.glob(os.path.join(loop_path, '*'))
            # Iterate over the files and remove those that don't match the extensions npy and mat
            for file in all_files:
                if file.endswith('.obj') or file.endswith('.ply') or file.endswith('.dat'):
                #not (file.endswith('.npy') or file.endswith('.mat') or file.endswith('.log') or file.endswith('.npz') or file.endswith('.png')):
                    os.remove(file)
        
    end = time.time()
    exp_time = end-start
    print("Timing of the experiment: {} sec".format(exp_time))

    return

def create_inputs(model1,model2,model_type, is_variables=True,is_delete=True,is_GT=True,is_randomTform=True,NdDownsample_num=None):
    inputs = {'goIcp': config_goIcp,
        'FastSpectrum1': [ config_FastSpectrum, 1],
        'FastSpectrum2': [ config_FastSpectrum, 2],
        'fmap': config_fmap,
        'analysis': config_analysis,
        'is_variables': is_variables,
        'model1': model1,
        'model2': model2, 
        'model_type': model_type,
        'is_delete': is_delete,
        'is_GT': is_GT,
        'is_randomTform': is_randomTform,
        'NdDownsample_num': NdDownsample_num
        } 
    return inputs

if __name__ == '__main__':

    inputs={}

    loops = 1

    import config_goIcp
    import config_FastSpectrum
    import config_fmap
    import config_analysis

    model_type = 'C1'
    model1 = '2019'
    model2 = '2020'
    is_variables=True # use your own variables in configuration
    is_delete=False # Delete unnecessary files used in multiple loops
    is_GT=False # if you insert GT data
    is_randomTform=False # if you need random transformation to model 2
    NdDownsample_num=None  #down sample models
    is_vis = True # show visual tranformations
    #inputs['is_vis']


    inputs = {'goIcp': config_goIcp,
        'FastSpectrum1': [ config_FastSpectrum, 1],
        'FastSpectrum2': [ config_FastSpectrum, 2],
        'fmap': config_fmap,
        'analysis': config_analysis,
        'is_variables': is_variables,
        'model1': model1,
        'model2': model2, 
        'model_type': model_type,
        'is_delete': is_delete,
        'is_GT': is_GT,
        'is_randomTform': is_randomTform,
        'NdDownsample_num': NdDownsample_num,
        'is_vis': is_vis
        } 
    main(loops, inputs)


