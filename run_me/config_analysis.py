import os, pathlib

def Get_Configuration(exp_path = '', my_model_type=None, my_type1=None, is_vis=None):
    
    config = {}
    #------------------------------------
    if my_model_type is None:
    # change as you like 
        config['model_type'] = 'C1' #'C1'
        config['type1'] = '2019' #'2019'
    else:
        config['model_type'] = my_model_type
        config['type1'] = my_type1

    config['type2'] = 'goIcp'
    config['k1'] = 30#24;#30;#30; #number of eigs for mesh1
    config['k2'] = 250#200 #236#238 #300 #120;#20; #number of eigs for mesh2
    config['k11'] = 200 #number of eigs for mesh1
    config['k22'] = 200 #236#300 #120;#20; #number of eigs for mesh2
    config['plot_diff'] = 0 # 1 for plotting the differences, 0 for no show
    #------------------------------------
    if is_vis is not None:
        config['plot_diff'] = is_vis
    else:
        config['plot_diff'] = 0
    #------------------------------------

    config['model1_path'] = os.path.join(exp_path,  config['model_type'] +  config['type1'] +'.ply')
    config['model2_path'] = os.path.join(exp_path,  config['model_type'] +  config['type2'] +'.ply')
    
    config['evecs1'] = os.path.join(exp_path, config['type1'] + '_eigvec.dat')
    config['evecs2'] = os.path.join(exp_path, config['type2'] + '_eigvec.dat')

    config['evals1'] = os.path.join(exp_path,  config['type1'] + '_eigval.dat')
    config['evals2'] = os.path.join(exp_path,  config['type2'] + '_eigval.dat')

    config['mass1'] = os.path.join(exp_path, config['type1'] + '_mass.dat')
    config['mass2'] = os.path.join(exp_path, config['type2'] + '_mass.dat')

    config['P12'] = os.path.join(exp_path, config['model_type'] + '_P12_corals.npz')
    config['P21'] = os.path.join(exp_path, config['model_type'] + '_P21_corals.npz')
    config['C21'] = os.path.join(exp_path, config['model_type'] + '_C21_corals.npy')
    config['C21_full'] = os.path.join(exp_path, config['model_type'] + '_C21_full_corals.npy')

    config['fig_singular_vals'] = os.path.join(exp_path, config['model_type'] + '_fig_singular_vals.pdf')
    config['fig_singular_vals_full'] = os.path.join(exp_path, config['model_type'] + '_fig_singular_vals_full.pdf')
    config['singular_vals'] = os.path.join(exp_path, config['model_type'] + '_singular_vals.npy')
    config['singular_vals_full'] = os.path.join(exp_path, config['model_type'] + '_singular_vals_full.npy')

    config['all_colors1'] = os.path.join(exp_path, config['model_type'] + 'all_colors1.npy')
    config['all_colors2'] = os.path.join(exp_path, config['model_type'] + 'all_colors2.npy')
    config['all_colors_titles'] = os.path.join(exp_path, config['model_type'] + 'all_colors_titles.npy')

    return config