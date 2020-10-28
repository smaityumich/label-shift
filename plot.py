import re
import pandas as pd
import numpy as np


def save_panda(filename, const = 0, labeled = True):

    exp = filename + '.txt'
    with open(exp) as fh:
        out = fh.read()

    out = re.split(r'\n', out)

    dict_list = []
    for d in out:
        try:
            dict_list.append(eval(d))
        except:
            continue
    df = pd.DataFrame(dict_list)
    
    if labeled:
        df = df.loc[df['labeled']==True]
        df['bayes_error'] = np.mean(df['bayes_error']) - const
        df['classical-excess'] = (df['classical'] - df['bayes_error']).astype('float32') 
        df['labeled-excess'] = (df['supervised'] - df['bayes_error']).astype('float32') 
        measure = ['classical-excess', 'labeled-excess']

    else:
        df = df.loc[df['labeled']==False]
        df['bayes_error'] = np.mean(df['bayes_error']) - const
        df['lipton-excess'] = (df['lipton'] - df['bayes_error']).astype('float32') 
        df['oracle-excess'] = (df['oracle'] - df['bayes_error']).astype('float32') 
        measure = ['lipton-excess', 'oracle-excess']

    agg_dict = dict()
   
    for key in measure:
        agg_dict[key] = ['mean', 'std', 'count']
    result = df.groupby(['n_source', 'n_target', 'setup'], as_index=False).agg(agg_dict)

    
    save_file = 'pickled_pds/' + filename + ('_labeled' if labeled else '_unlabeled') + '.pkl'
    result.to_pickle(save_file)
    return result


result = save_panda('res', const = 0.039)
result1, result2, result3 = result.loc[result['setup'] == 'const'],\
     result.loc[result['setup'] == 'dec_sqrt'], result.loc[result['setup'] == 'dec_linear']
res1, res2, res3 = result1.loc[result1['n_target'] == 40],\
     result2.loc[result2['n_target'] == 40], result3.loc[result3['n_target'] == 40]