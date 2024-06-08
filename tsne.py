import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt_sne
from sklearn import datasets
from sklearn.manifold import TSNE
import os
 
 
def plot_tsne(features, labels, epoch,fileNameDir = None):
    '''
    features:(N*m) N*m大小特征，其中N代表有N个数据，每个数据m维
    label:(N) 有N个标签
    '''
    print(features.shape,labels.shape)
    print(type(features),type(labels))
    print(np.any(np.isnan(features)),np.any(np.isinf(features)))
    features = np.nan_to_num(features)
    if not os.path.exists(fileNameDir):
        os.makedirs(fileNameDir)
    import pandas as pd
    tsne = TSNE(n_components=2, init='pca', random_state=0)
    import seaborn as sns
    
    #查看标签的种类有几个
    class_num = len(np.unique(labels))  # 要分类的种类个数  eg:[0, 1, 2, 3]这个就是为4
   
    try:
        tsne_features = tsne.fit_transform(features)  # 将特征使用PCA降维至2维
    except:
        tsne_features = tsne.fit_transform(features)
    
    #一个类似于表格的数据结构
    df = pd.DataFrame()
    df["y"] = labels
    df["comp1"] = tsne_features[:, 0]
    df["comp2"] = tsne_features[:, 1]
 
    hex = ["#ff8b8b", "#6e85b7"]
    data_label = []
    for v in df.y.tolist():
        if v == 1:
            data_label.append("one")
        else:
            data_label.append("zero")
    df["value"] = data_label
    
    # hue:根据y列上的数据种类，来生成不同的颜色；
    # style:根据y列上的数据种类，来生成不同的形状点；
 
    sns.scatterplot(x= df.comp1.tolist(), y= df.comp2.tolist(),hue=df.value.tolist(),style = df.value.tolist(),
                    palette=sns.color_palette(hex,class_num),markers= {"one":"v","zero":"^"},
                    data=df).set(title="T-SNE projection")
   
    plt_sne.savefig(os.path.join(fileNameDir,"%s.jpg") % str(epoch),format = "jpg")
    plt_sne.show()
 
 
if __name__ == '__main__':
    digits = datasets.load_digits(n_class=2)
    features, labels = digits.data, digits.target
    print(features.shape)
    print(labels.shape)
 
    plot_tsne(features, labels, "Set2", fileNameDir="test")