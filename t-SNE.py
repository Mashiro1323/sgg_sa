from sklearn import datasets
from openTSNE import TSNE
 
import tsneutil
 
import matplotlib.pyplot as plt
iris = datasets.load_breast_cancer()
x, y = iris["data"], iris["target"]
 
 
tsne = TSNE(
    perplexity=50,
    n_iter=500,
    metric="euclidean",
    # callbacks=ErrorLogger(),
    n_jobs=8,
    random_state=42,
)
embedding = tsne.fit(x)
tsneutil.plot(embedding, y, colors=tsneutil.MOUSE_10X_COLORS)
 
 
print('end')