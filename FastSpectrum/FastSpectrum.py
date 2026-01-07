import sys
sys.path.append('../shape_analysis_system')

import subprocess 
import os
import time

#def main(model_num, config_FastSpectrum, exp_directory=''):
def main(config):
    
    start = time.time()
    #config = config_FastSpectrum.Get_Configuration(model_num, exp_directory)

    type_files = []
    type_files.append('eigvec')
    type_files.append('eigval')
    type_files.append('mass')
    
    files = []
    files.append(os.path.join(config['saveDest'], config['type'] + '_' + type_files[0] + '.dat'))
    files.append(os.path.join(config['saveDest'], config['type'] + '_' + type_files[1] + '.dat'))
    files.append(os.path.join(config['saveDest'], config['type'] + '_' + type_files[2] + '.dat'))

    # remove old files
    for file in files:
        if os.path.exists(file):
            os.remove(file)

    args = [config['exe_path'], config['meshFile'], config['saveDest'], config['type'], config['num_sampels']]
    # Open the log file
    with open(config['log_path'] + config['type'] + '.log', 'w') as log_file:
        # Run the executable and redirect its output to the log file                                                           
        return_code = subprocess.run(args, stdout=log_file, stderr=log_file)
    end = time.time()
    total_time = end-start
    if return_code.returncode:
        print('error')
    else:
        print('success')

    print(f'Runtime FastSpectrum {total_time} sec.')
    return total_time

if __name__ == '__main__':

    import config as config_FastSpectrum
    config1 = config_FastSpectrum.Get_Configuration(1)
    config2 = config_FastSpectrum.Get_Configuration(2)
    main(config1) #first mesh
    main(config2) #second mesh