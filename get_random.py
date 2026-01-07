#import matplotlib
#matplotlib.use('TKAgg')
import numpy as np
import copy
import open3d as o3d
import os
import pathlib


def main(model_type, type2):
    curr_path = pathlib.Path(__file__).parent.resolve() 
    data_path = os.path.join(curr_path, 'data')
    model2_path = os.path.join(data_path, model_type, model_type + type2 + '.ply') #model after changes
    randomTform_path = os.path.join(data_path, model_type,model_type + '_randomTform.npy')
    model2R_path = os.path.join(data_path,model_type, model_type + type2 + '_R.ply') 
    

    #read the model
    pData_raw = o3d.io.read_triangle_mesh(filename=str(model2_path))
    #------------------------------------------
    # Load the random transformation
    print("Random transformation")
    #load random transformation
    randTform = np.load(randomTform_path,allow_pickle=True)
    pDataRT =copy.deepcopy(pData_raw).transform(randTform)

    #save mesh
    o3d.io.write_triangle_mesh(model2R_path, pDataRT, write_ascii=True)

if __name__ == '__main__':

    model_type = 'C1'
    #type1 = '2019'
    type2 = '2020'

    paths = main(model_type, type2)
            


