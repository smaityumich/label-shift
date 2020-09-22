import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn import metrics


def lipton_method(x_source, y_source, x_target):
    

    x_source = np.array(x_source)
    x_target = np.array(x_target)
    y_source = np.array(y_source)
    m, _ = np.shape(x_source)
    prop_source = np.mean(y_source)

    cl = LogisticRegression(penalty='none')
    cl.fit(x_source, y_source)
    confusion_matrix = metrics.confusion_matrix(cl.predict(x_source),y_source)/m
    prop_target = np.mean(cl.predict(x_target))
    
    xi = np.array([1-prop_target,prop_target])
    w = np.matmul(np.linalg.inv(confusion_matrix),xi)
    prop_targets = w*np.array([1-prop_source, prop_source])
    prop_targets = prop_targets/np.sum(prop_targets)
    return prop_targets[1], w


if __name__ == "__main__":
    pass




