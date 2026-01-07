import sys
sys.path.append('../shape_analysis_system')
import utils
#import config as configuration

import numpy as np
#matplotlib.use('TKAgg')
from scipy.sparse import csr_matrix
import matplotlib.pyplot as plt
import time
import scipy
import os
import re  # Regular expressions for extracting numbers

def get_max_idx(titles):
    # Extract 's' values and convert them to floats
    s_values = [float(re.search(r"s = (\d+\.\d+)", desc).group(1)) for desc in titles]

    # Find indices where s >= 1
    valid_indices = [i for i, s in enumerate(s_values) if s >= 1]

    # Get the maximum index from the list of valid indices
    return max(valid_indices) if valid_indices else None  # Returns None if the list is empty
    


def showMapSvd_corals(mesh1, basis1, mesh2, basis2, C21, P12, P21, numEig, model_type, types, fig_singular_vals_path,singular_vals_path,plot_diff=False, vistype=1):
    #P12 -> n1xn2 mapping from mesh1 to mesh2 (vertex 1 -> vertex 2)
    #P21 -> n2xn1 mapping from mesh2 to mesh1 (vertex 2 -> vertex 1)
    #basis1 -> n1xk1
    #basis2 -> n2xk2
    #C21 -> k2xk1 fmap from mesh1 to mesh2
    #numEig is max number of r from paper??
    #vistype: 1= , 
    # 2= screen col2 on col1 - mesh2 distortions on mesh1 (pmap n1x1), 
    # 3= screen col1 on col2 - mesh1 distortions on mesh2 (pmap n2x1)

    print('Plot Singular Values')
    U21,S21,V21  = np.linalg.svd(C21)
    #V-> k1xk1
    #U -> k2xk2

    # plot Singular values
    plt.plot(S21)
    plt.title('S21 - Singular Values')
    plt.xlabel('k')
    plt.ylabel('Singular Value')
    plt.axhline(y = 1, color = 'r', linestyle = '-')
    plt.savefig(fig_singular_vals_path, dpi=300)
    plt.close()
    np.save(singular_vals_path, S21)
    #plt.show()

    functions1 = []
    functions2 = []
    titles = []
    #x= numEig
    #first_ten = np.arange(10)
    #last_ten = np.arange(x-10, x)
    #combined_array = np.concatenate((first_ten, last_ten))
    for i in np.arange(numEig):
        #v1 = basis1.dot(V21[:,i][:, np.newaxis]) 
        v1 = basis1.dot(V21[:,i]) #v1 is a vector
        col1 = csr_matrix(v1**2).T #why? sparse

        u1 = basis2.dot(U21[:,i]) #u1 is a vector
        col2 = csr_matrix(u1**2).T #why? sparse

        if vistype == 2:
            col1 = P12.dot(col2) #screen col2 on col1 - mesh2 distortions on mesh1 (pmap n1x1)
    
        if vistype == 3:
            col2 = P21.dot(col1) #screen col1 on col2 - mesh1 distortions on mesh2 (pmap n2x1)

        #plot the diffrences
        colors1 = col1.toarray().flatten()
        colors2 = col2.toarray().flatten()
        title = f"{model_type}, k = {i+1}, s = {round(S21[i], 4)}"
        #my_k=[1,5,10,13,22,26,28]
        #if plot_diff and (i+1) in my_k:
        #if plot_diff:
        if plot_diff and (i+1) > 25:
            utils.pv_compare_analysis(mesh1, mesh2, title, types, [colors1,colors2])
        functions1.append(colors1)
        functions2.append(colors2)
        titles.append(title)
    
    #normalize
    functions1 = [(f - np.min(f)) / (np.max(f) - np.min(f)) if np.max(f) != np.min(f) else f for f in functions1]
    functions2 = [(f - np.min(f)) / (np.max(f) - np.min(f)) if np.max(f) != np.min(f) else f for f in functions2]

# split according to s value - idx of the last number that big or equal to 1
    idx_bigger_1 = get_max_idx(titles)

    if idx_bigger_1 is None:
        # take the max value of each vertex
        all_colors1 = np.max(np.stack(functions1), axis=0)
        all_colors2 = np.max(np.stack(functions2), axis=0) 
        if plot_diff:
            utils.pv_compare_analysis(mesh1, mesh2, "Changes with s>=1", types, [all_colors1,all_colors2])
    else:
        idx_bigger_1+=1
        functions1_big = functions1[:idx_bigger_1]
        functions1_small = functions1[idx_bigger_1:]

        functions2_big = functions2[:idx_bigger_1]
        functions2_small = functions2[idx_bigger_1:]

        # take the max value of each vertex
        all_colors1 = np.max(np.stack(functions1_big), axis=0)
        all_colors2 = np.max(np.stack(functions2_big), axis=0) 
        if plot_diff:
            utils.pv_compare_analysis(mesh1, mesh2, "Changes with s>=1", types, [all_colors1,all_colors2])

        all_colors11 = np.max(np.stack(functions1_small), axis=0)
        all_colors22 = np.max(np.stack(functions2_small), axis=0) 
        if plot_diff:
            utils.pv_compare_analysis(mesh1, mesh2, f"Changes with s<1", types, [all_colors11,all_colors22])

    return functions1, functions2, titles 

    

#def main(config_analysis, exp_directory=''):
def main(config):
    
    # Analyze a map from mesh1 to mesh2
    #T21: 2->1
    #1 2019
    #2 2020
    start = time.time()
   #config = config_analysis.Get_Configuration(exp_directory)

    # Load the meshes
    print('Load Meshes')
    mesh1 = utils.read_ply_file(config['model1_path'])
    n1 = mesh1['vertices'].shape[0]
    mesh2 = utils.read_ply_file(config['model2_path'])
    n2 = mesh2['vertices'].shape[0]
    #n1>n2

    #eigs
    print('Read eigenvectors')
    basis1 = utils.read_binary_matVec(config['evecs1'])
    #basis1 = np.load(config['evecs1'],allow_pickle=True)
    basis2 = utils.read_binary_matVec(config['evecs2'])
    #basis2 = np.load(config['evecs2'],allow_pickle=True)
    
    if basis1.shape[1] < config['k1']:
        config['k1'] = basis1.shape[1]
        print("k1 was changed to: {}".format(config['k1']))
    if basis2.shape[1] < config['k2']:
        config['k2'] = basis2.shape[1]
        print("k2 was changed to: {}".format(config['k2']))

    #k1<k2
    numEig1 = config['k1']
    numEig2 = config['k2']  #k2 Use more eigens for second shape
    numEig_full = config['k11']

    basis11=basis1[:,:numEig1]
    basis22=basis2[:,:numEig2]

    #pmap+fmap:

    # Load the map. Note, assumes 0-based.
    print('Load Maps')
    P12 = scipy.sparse.load_npz(config['P12']).tocsr()
    P21 = scipy.sparse.load_npz(config['P21']).tocsr()

    # Convert map to fmap
    C21 = np.load(config['C21'],allow_pickle=True)
    C21_full = np.load(config['C21_full'],allow_pickle=True)

    # Show first and last singular vectors
    # screening 2 on 1
    print('Show Map SVD')
    types = [config['type1'], config['type2']]
    
    #if config['plot_diff']:
    functions1,functions2, titles = showMapSvd_corals(mesh1, basis11, mesh2, basis22, C21, P12, P21, numEig1, 
                    config['model_type'], types, config['fig_singular_vals'] ,config['singular_vals'], config['plot_diff'], vistype=2)
    np.save(config['all_colors1'], functions1)
    np.save(config['all_colors2'], functions2)
    np.save(config['all_colors_titles'], titles)
    """
    else:
        print('Plot Singular Values')
        _,S21,_  = np.linalg.svd(C21)
        #V-> k1xk1
        #U -> k2xk2

        # plot Singular values
        plt.plot(S21)
        plt.title('S21 - Singular Values')
        plt.xlabel('k')
        plt.ylabel('Singular Value')
        plt.axhline(y = 1, color = 'r', linestyle = '-')
        plt.savefig(config['fig_singular_vals'] , dpi=300)
        plt.close()
        np.save(config['singular_vals'], S21)
        #plt.show() 
        """

    _,S21_full,_  = np.linalg.svd(C21_full)
    #V-> k1xk1
    #U -> k2xk2

    # plot Singular values
    plt.plot(S21_full)
    plt.title('S21 full - Singular Values')
    plt.xlabel('k')
    plt.ylabel('Singular Value')
    plt.axhline(y = 1, color = 'r', linestyle = '-')
    plt.savefig(config['fig_singular_vals_full'] , dpi=300)
    plt.close()
    np.save(config['singular_vals_full'], S21_full) 

    
    end = time.time()
    total_time = end-start
    return total_time

if __name__ == '__main__':
    
    import config as config_analysis
    import pathlib

    model_type= 'C3'
    model1= '2020'
    model2= '2022'
    is_GT =  'not_GT'
    parent_path = pathlib.Path(__file__).parent.resolve().parent.absolute()
    path = os.path.join(parent_path,'experiments','results',is_GT,model_type +'_'+model1+'_'+model2,'0')
    #path = os.path.join(parent_path,'experiments','results',model_type +'_'+model1+'_'+model2,'0')
    config = config_analysis.Get_Configuration(path, model_type, model1)
    main(config)