import matplotlib
matplotlib.use('TKAgg')
import numpy as np
import pathlib
#from goIcp import goIcp_utils #when i run from ex_main
import sys
import os
import pymeshlab
sys.path.append(os.path.join(pathlib.Path(__file__).parent.resolve().parent.absolute())) # main project path

import time
import glob

def manifold(model_path, manifold_path, log_path, dir):
    print(f'Start manifold process for {dir}')
    ms = pymeshlab.MeshSet()
    ms.load_new_mesh(model_path)
    topo_1 = ms.get_topological_measures()
    ms.meshing_repair_non_manifold_edges()
    ms.meshing_repair_non_manifold_vertices()
    ms.meshing_remove_connected_component_by_diameter()
    #topological measures
    topo_2 = ms.get_topological_measures()

    # save topological measures to log (before and after)
    with open(log_path, 'w') as f:
        f.write('Before:\n')
        for key, value in topo_1.items(): 
            f.write('%s:%d\n' % (key, value))
        f.write('\nAfter:\n')
        for key, value in topo_2.items(): 
            f.write('%s:%d\n' % (key, value))

    #check topological measures before and after repair
    print(f'Check topological measures after the repair in model {dir}')
    if topo_2['connected_components_number'] != 1:
        print(f'There is more than 1 connected component in mesh {dir}')
        return False
    if topo_2['is_mesh_two_manifold'] != True:
        print(f'Mesh {dir} is not manifold')
        return False

    #save the repaired mesh - only if it is manifold
    ms.save_current_mesh(file_name=manifold_path, binary=False, save_vertex_quality=False,
                        save_vertex_radius=False, save_face_quality=False, save_face_color=False,
                        save_wedge_color=False, save_wedge_normal=False)
    # clear the mesh set after saving
    ms.clear()
    return True

#  exp_directory - directory to save experimrnts
def main(main_path, dst_path, model_type):
    # Open a file where you want to log the prints
    main_log_file = open(os.path.join(main_path, 'log2manifold.log'), 'w')
    # Redirect print output to the log file
    sys.stdout = main_log_file
    
    start = time.time()
    all_dirs = [d for d in glob.glob(os.path.join(main_path, '*'))]
    for  dir in all_dirs:
        dir_name =  os.path.basename(dir)
        #if os.path.isdir(dir) and ( dir_name == 'C5'):
        if os.path.isdir(dir) and (dir_name == 'NR1'):
            model_path = os.path.join( dir, dir_name + model_type +'.ply')
            manifold_path = os.path.join(dst_path,  dir_name, dir_name + model_type +'_manifold.ply')
            log_path = os.path.join(dst_path,  dir_name ,'log_' + dir_name + model_type + '.txt')
            assert manifold(model_path, manifold_path, log_path, dir_name), "The Models are not manifold"
    end = time.time()-start
    print(f'Run for {end} sec')
    
    # Reset sys.stdout to its default value if needed
    sys.stdout = sys.__stdout__
    return 

#  exp_directory - directory to save experimrnts
def main2(main_path, dst_path, model_type):
    # Open a file where you want to log the prints
    main_log_file = open(os.path.join(main_path, 'log2manifold.log'), 'w')
    # Redirect print output to the log file
    sys.stdout = main_log_file
    
    start = time.time()
    model_path = os.path.join( main_path, model_type +'.ply')
    manifold_path = os.path.join(dst_path, model_type +'_manifold.ply')
    log_path = os.path.join(dst_path, 'log_' + model_type + '.txt')
    #assert manifold(model_path, manifold_path, log_path, model_type), "The Model is not manifold"
    manifold(model_path, manifold_path, log_path, model_type)
    end = time.time()-start
    print(f'Run for {end} sec')
    
    # Reset sys.stdout to its default value if needed
    sys.stdout = sys.__stdout__
    return 


if __name__ == '__main__':
    
    #model_type = 'skull_83595'
    #model_type = 'skull_83596'
    model_type = 'Chunk1IUI24-002'
    dir = os.path.join('Chunk1IUI24')

    curr_path = pathlib.Path(__file__).parent.resolve() #directory of the script being run
    main_path = os.path.join(curr_path.parent.absolute(), 'data','not_manifold',dir)
    dst_path = os.path.join(curr_path.parent.absolute(), 'data','manifold',dir)

    #main(main_path, dst_path, model_type)
    main2(main_path, dst_path, model_type)
