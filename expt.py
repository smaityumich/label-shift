import numpy as np
import setup
import data
import sys
import os




## For filename don't provide extension. It will be set to .out

def f(n_source, n_target, prop_target, prop_source = 0.5,\
     labeled = True, d = 3, distance = 1, kernel_df = 3, beta = 3, iteration = 0):
    
    D = data.DataGenerator(d = d)
    x_source, y_source = D.getData(n_source, prop_source, distance=distance)
    x_target, y_target = D.getData(n_target, prop_target, distance=distance)
    x_test, y_test = D.getData(100, prop_target, distance=distance)
    bayes_error = D.bayes_error(prop=prop_target, distance=distance)
    parameter = beta, kernel_df, prop_target
    return_dict = setup.excess_risk(parameter, x_source, y_source, x_target,\
         y_target, x_test, y_test, bayes_error, labeled=labeled)

    return_dict['iter'] = iteration
    return return_dict


if __name__ == "__main__":

    if not os.path.exists('temp/'):
        os.mkdir('temp/')


    beta = 2
    sample_sizes = np.load('sample_sizes.npy')
    label = np.load('label.npy')
    #print('Done')
    i = int(sys.argv[1])
    filename = f'{i}.txt'
    iteration = i % 100

    j = i // 100
    k = j % 34
    n_source, n_target = sample_sizes[k, 0], sample_sizes[k, 1]
    labeled = label[k]

    #j = j // 17
    #beta = betas[j]
    kernel_df = beta
    print(f'n_s: {n_source}, n_t: {n_target}, kernel_df: {kernel_df}, beta: {beta}, iter: {iteration}, label: {labeled}')
    prop_source = 10/(n_source)

    if (labeled == False) and (n_source == 1000):
        m = 10
    else:
        m = 1

    print(m)
        
    return_dict = []
    for _ in range(m):
        if n_target == 100:
            prop_source1, prop_source2, prop_source3 = 0.5, 0.5 * np.sqrt(20/n_source), 0.5 * 20 / n_source
        elif n_target == 40: 
            prop_source1, prop_source2, prop_source3 = 0.5, 0.5 * np.sqrt(25/n_source), 0.5 * 25 / n_source
        else: 
            prop_source1, prop_source2, prop_source3 = 0.5, 0.5, 0.5


        return_dict1 = f(int(n_source), int(n_target), 0.75, prop_source= prop_source1, labeled=labeled, kernel_df=int(kernel_df), beta= beta,\
                    iteration=int(iteration), distance=2)
        return_dict1['setup'] = 'const'
        return_dict.append(return_dict1)


            
        return_dict1 = f(int(n_source), int(n_target), 0.75, prop_source= prop_source2, labeled=labeled, kernel_df=int(kernel_df), beta= beta,\
                    iteration=int(iteration), distance=2)
        return_dict1['setup'] = 'dec-sqrt'
        return_dict.append(return_dict1)

            
        return_dict1 = f(int(n_source), int(n_target), 0.75, prop_source= prop_source3, labeled=labeled, kernel_df=int(kernel_df), beta= beta,\
                    iteration=int(iteration), distance=2)
        return_dict1['setup'] = 'dec-linear'
        return_dict.append(return_dict1)
    
    print(return_dict)


    with open('temp/const_' + filename, 'a') as f:
        for r in return_dict:
            f.writelines(str(r)+"\n")
