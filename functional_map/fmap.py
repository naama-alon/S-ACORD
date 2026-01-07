import sys
sys.path.append('../shape_analysis_system')
import utils

import numpy as np
import scipy
import matplotlib.pyplot as plt
from sklearn.neighbors import NearestNeighbors
import time


def find_P21_mapping(verts1, verts2 , n1, n2):
    nbrs = NearestNeighbors(n_neighbors=1).fit(verts1)
    distances, pmap2 = nbrs.kneighbors(verts2) # n2x1 map from mesh2 to mesh1
    #[pmap2, dist2] = knnsearch(verts1,verts2) 
    i = np.arange(n2)
    j = np.squeeze(pmap2.transpose())
    v = np.ones([n2])
    P21 = scipy.sparse.coo_matrix((v,(i,j)), shape=(n2,n1)) #n2xn1 from mesh1 to mesh2
    return P21

#def main(exp_directory=''):
def main(config):

    start = time.time()
    #config = config_fmap.Get_Configuration(exp_directory)
    
    ## Get meshes and eigenvectors
    #k1>k2 for 21


    mesh1 = utils.read_ply_file(config['model1_path'])
    n1 = mesh1['vertices'].shape[0]
    mesh2 = utils.read_ply_file(config['model2_path'])
    n2 = mesh2['vertices'].shape[0]

    mass_matrix2_tmp = utils.read_binary_matVec(config['mass2'])
    mass_matrix2 = scipy.sparse.spdiags(mass_matrix2_tmp[:,0], diags=0, m=n2, n=n2)

    psi1 = utils.read_binary_matVec(config['evecs1'])
    psi2 = utils.read_binary_matVec(config['evecs2'])
    

    if psi1.shape[1] < config['k1']:
        config['k1'] = psi1.shape[1]
        print("k1 was changed to: {}".format(config['k1']))
    if psi2.shape[1] < config['k2']:
        config['k2'] = psi2.shape[1]
        print("k2 was changed to: {}".format(config['k2']))

    #for full
    psi11 = psi1[:,:config['k11']]
    psi22 = psi2[:,:config['k22']]
    psi22_T = psi22.transpose()*mass_matrix2
    # Check that the basis is orthonormal with respect to the area matrix.
    assert np.linalg.norm((psi22_T.dot(psi22) - np.eye(config['k22'],config['k22'])),'fro') < 1e-7
    
    psi1 = psi1[:,:config['k1']]
    psi2 = psi2[:,:config['k2']]
    psi2_T = psi2.transpose()*mass_matrix2
    # Check that the basis is orthonormal with respect to the area matrix.
    assert np.linalg.norm((psi2_T.dot(psi2) - np.eye(config['k2'],config['k2'])),'fro') < 1e-7


    
    ## find mapping & functional map P21,C21
    print('Get P21')
    # P21 - from mesh1 to mesh2
    P21 = find_P21_mapping(mesh1['vertices'], mesh2['vertices'] , n1, n2)
    #show P21
    if config['isVis']:
        plt.spy(P21, markersize = 1)
        plt.ticklabel_format(style='sci', axis='y', scilimits=(0,0))
        plt.title('P21')
        plt.show()

    print('Get C21')
    # C21 - from mesh1 to mesh2
    C21 = psi2_T.dot(P21.dot(psi1)) #k2xk1
    C21_full = psi22_T.dot(P21.dot(psi11)) #k22xk11
    # show functional map
    #if config['isVis']:
    ax = plt.subplot()
    im = ax.imshow(C21, aspect='auto')
    plt.title('C21')
    plt.colorbar(im)
    #plt.show()
    plt.savefig(config['fig_C21'] , dpi=300)
    plt.close()

    ax = plt.subplot()
    im = ax.imshow(C21_full, aspect='auto')
    plt.title('C21 Full')
    plt.colorbar(im)
    #plt.show()
    plt.savefig(config['fig_C21_full'] , dpi=300)
    plt.close()


    ## find mapping P12
    print('Get P12')
    # P12 - from mesh2 to mesh1
    P12 = find_P21_mapping(mesh2['vertices'], mesh1['vertices'] , n2, n1)
    #show P12
    if config['isVis']:
        plt.spy(P12, markersize = 1)
        plt.ticklabel_format(style='sci', axis='x', scilimits=(0,0))
        plt.title('P12')
        plt.show()  

    #save
    print('Save')
    #np.save(config['P21'], P21)
    scipy.sparse.save_npz(config['P21'], P21)
    #savemat(config['P21_mat'], {'P21': P21}) #save in mat
    np.save(config['C21'], C21)
    np.save(config['C21_full'], C21_full)
    #np.save(config['P12'], P12)
    scipy.sparse.save_npz(config['P12'], P12)

    end = time.time()
    total_time = end-start

    return total_time



if __name__ == '__main__':
    #import config as config_fmap
    #config = config_fmap.Get_Configuration()
    #main(config)
    import config_new as config_fmap
    import pathlib
    import os

    model_type= 'C3'
    model1= '2020'
    model2= '2022'
    is_GT = 'not_GT'
    parent_path = pathlib.Path(__file__).parent.resolve().parent.absolute()
    path = os.path.join(parent_path,'experiments','results',is_GT,model_type +'_'+model1+'_'+model2,'0')
    #path = os.path.join(parent_path,'experiments','results',model_type +'_'+model1+'_'+model2,'0')
    config = config_fmap.Get_Configuration(path, model_type, model1)
    main(config)