import numpy as np
import setup
import data
import sys
import os




## For filename don't provide extension. It will be set to .out

def f(n_source, n_target, prop_target, prop_source = 0.5,\
     labeled = True, d = 4, distance = 1, kernel_df = 3, beta = 3, iteration = 0):
    
    D = data.DataGenerator(d = d)
    x_source, y_source = D.getData(n_source, prop_source, distance=distance)
    x_target, y_target = D.getData(n_target, prop_target, distance=distance)
    x_test, y_test = D.getData(1000, prop_target, distance=distance)
    bayes_error = D.bayes_error(prop=prop_target, distance=distance)
    parameter = beta, kernel_df, prop_target
    return_dict = setup.excess_risk(parameter, x_source, y_source, x_target,\
         y_target, x_test, y_test, bayes_error, labeled=labeled)

    return_dict['iter'] = iteration
    return return_dict


if __name__ == "__main__":

    if not os.path.exists('temp/'):
        os.mkdir('temp/')


    beta = 3
    sample_sizes = np.load('sample_sizes.npy')
    label = np.load('label.npy')
    print('Done')
    i = int(sys.argv[1])
    filename = f'{i}.txt'
    iteration = i % 100

    j = i // 100
    k = j % 28
    n_source, n_target = sample_sizes[k, 0], sample_sizes[k, 1]
    labeled = label[k]

    #j = j // 17
    #beta = betas[j]
    kernel_df = beta
    print(f'n_s: {n_source}, n_t: {n_target}, kernel_df: {kernel_df}, beta: {beta}, iter: {iteration}, label: {labeled}')

    return_dict1 = f(int(n_source), int(n_target), 0.75, labeled=labeled, \
        kernel_df=int(kernel_df), beta= beta,\
             iteration=int(iteration), distance=2)

    

    return_dict1['setup'] = 'constant-prop-source'
    print(return_dict1)


    with open('temp/const_' + filename, 'w') as f:
        f.writelines(str(return_dict1)+"\n")
