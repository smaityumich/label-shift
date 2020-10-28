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




def multicolor_ylabel(ax,list_of_strings,list_of_colors,axis='x',anchorpad=0,**kw):
    """this function creates axes labels with multiple colors
    ax specifies the axes object where the labels should be drawn
    list_of_strings is a list of all of the text items
    list_if_colors is a corresponding list of colors for the strings
    axis='x', 'y', or 'both' and specifies which label(s) should be drawn"""
    from matplotlib.offsetbox import AnchoredOffsetbox, TextArea, HPacker, VPacker

    # x-axis label
    if axis=='x' or axis=='both':
        boxes = [TextArea(text, textprops=dict(color=color, ha='left',va='bottom',**kw)) 
                    for text,color in zip(list_of_strings,list_of_colors) ]
        xbox = HPacker(children=boxes,align="center",pad=0, sep=5)
        anchored_xbox = AnchoredOffsetbox(loc=3, child=xbox, pad=anchorpad,frameon=False,bbox_to_anchor=(0.2, -0.09),
                                          bbox_transform=ax.transAxes, borderpad=0.)
        ax.add_artist(anchored_xbox)

    # y-axis label
    if axis=='y' or axis=='both':
        boxes = [TextArea(text, textprops=dict(color=color, ha='left',va='bottom',rotation=90,**kw)) 
                     for text,color in zip(list_of_strings[::-1],list_of_colors) ]
        ybox = VPacker(children=boxes,align="center", pad=0, sep=5)
        anchored_ybox = AnchoredOffsetbox(loc=3, child=ybox, pad=anchorpad, frameon=False, bbox_to_anchor=(-0.10, 0.2), 
                                          bbox_transform=ax.transAxes, borderpad=0.)
        ax.add_artist(anchored_ybox)