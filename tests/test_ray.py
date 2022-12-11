# This script is to test the analysis results of DELTA-ray verison with the results of DELTA-mpi version. 
# required arguments: 
# --dir_1:   directory where DELTA-Ray analysis results are stored
# --dir_2:   directory where DELTA-mpi analysis results are stored
# --task_name:  analysis task name ex. task_crosscorr, task_crosscorr_cu   

# Example: 
# To run the test, use the following command line inside that runs inside DELTA shifter image: 

# shifter pytest -s test_ray.py --dir_1=/pscratch/sd/m/maburidi/data_storage/data_storage_ray_delta/ --dir_2=/pscratch/sd/m/maburidi/data_storage/data_storage_old_delta/ --task_name=task_crosscorr --num_chunks=100

import numpy as np

def test_results(params_ray):
    storgae_dir1 = params_ray['dir_1'] 
    storgae_dir2 = params_ray['dir_2'] 
    task_name = params_ray['task_name'] 
    num_chunks = int(params_ray['num_chunks'])

    for i in range(num_chunks):
        stri = "00000"
        chunk_number = str(i)
        chunk_id= stri[:5-len(chunk_number)] + str(chunk_number) 
        
        datafile_name1 = task_name+ "_chunk" + chunk_id + "_batch00.npz" 
        datafile_name2 = task_name+ "_chunk" + chunk_id + "_batch00.npz" 
        
        data_ray = np.load(storgae_dir1 + datafile_name1)
        data_old = np.load(storgae_dir2 + datafile_name2)
        
        assert np.allclose(data_old["arr_0"], data_ray["arr_0"], rtol=1e-15, atol=1e-08, equal_nan=True) == True
        msg = "Chunk " + str(i) + " Passed"
        #print(msg)