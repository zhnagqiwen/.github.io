# 必要なライブラリをインポート
import pandas as pd
%matplotlib inline
import matplotlib.pyplot as plt

# CSVファイルを読み込む
df = pd.read_csv("2016-2019.csv", index_col=0)
df

# 階層型クラスタリング用のライブラリをインポート
from scipy.cluster.hierarchy import linkage, dendrogram, fcluster

# データフレームから必要なデータを選択
X = df.iloc[:, 0:2].values

# ウォード法とユークリッド距離を使用してリンケージを計算
linkage_result = linkage(X, method='ward', metric='euclidean')

# デンドログラムをプロット
plt.figure(num=None, figsize=(6, 3), dpi=200, facecolor='w', edgecolor='k')
dendrogram(linkage_result, labels=df.index)
plt.show()

# 階層型クラスタリングモデルをインポート
from sklearn.cluster import AgglomerativeClustering

# モデルのトレーニング
hir_clus   = AgglomerativeClustering(n_clusters = 3, affinity = 'euclidean', linkage = 'ward')
y_hir_clus1 = hir_clus.fit_predict(X)
y_hir_clus1
df["Class1"] = y_hir_clus1
df

# クラスターの配列情報
cluster_labels = np.unique(y_hir_clus1)  # 一意なクラスター要素
n_clusters = cluster_labels.shape[0]    # 配列の長さ

# 可視化
for i in range(len(cluster_labels)):
    color = cm.jet(float(i) / n_clusters)
    plt.scatter(X[y_hir_clus1 == i, 0], X[y_hir_clus1 == i, 1], s = 50, c = color, label = 'Cluster'+str(i))

plt.title('銀行のクラスター')
plt.xlabel('NII')
plt.ylabel('NNI')
plt.legend(loc="best")
plt.show()

# 別のデータセットでクラスタリングを行う
X1 = df.iloc[:,0:3].values
linkage_result = linkage(X1, method='ward', metric='euclidean')
plt.figure(num=None, figsize=(6, 3), dpi=200, facecolor='w', edgecolor='k')
dendrogram(linkage_result, labels=df.index)
plt.show()
hir_clus2   = AgglomerativeClustering(n_clusters = 4, affinity = 'euclidean', linkage = 'ward')
y_hir_clus2 = hir_clus2.fit_predict(X1)
y_hir_clus2
df["Class2"] = y_hir_clus2
df

# 結果をCSVファイルに保存
df.to_csv("Bank_Cluster_QE4t.csv")
