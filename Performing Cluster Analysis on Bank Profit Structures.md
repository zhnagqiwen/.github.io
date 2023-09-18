"""
特異な利益構造を持つ銀行の特定や、分類された利益構造別に利益率との関係を見出すことが可能かを検証するために、全国銀行協会が公表している財務諸表の財務データを用いて、利益構造別に銀行をクラスター分析する。サンプルは2010年度から2019年度の日本の都市銀行と地方銀行を観測対象とする 。また、単年度による「ぶれ」の影響を解消するために、データは各量的緩和策期間に対応する値（2010年度～2012年度、2013年度～2015年度、2016年度～2019年度）それぞれの平均値を用いた。その中，〖QE2〗_tは包括的金融緩和期（2010年度～2012年度）、〖QE3〗_tは量的・質的金融緩和期（2013年度〜2015年度）、〖QE4〗_tはマイナス金利付き量的・質的金融緩和（2016 年度〜2019 年度）を示す。また、各銀行の元の財務データの金額のままでは、グループ分けは銀行の規模によって決まる可能性があり、それを防ぐために、財務諸表の各勘定項目の値は各行ごとに総資産残高で標準化する。
本分析で銀行の利益構造の特徴を定量化して、似ている指標・変数ごとに分類していく過程で「樹形図（デンドログラム）」を出力できる。階層的方法でクラスター(階層クラスター分析) の分割を行う。また、分析の主眼として利益構造が直接に影響力をもつように特徴量を選択しなければならないから、本論文では銀行の利益構造の部分構成を特徴量とする。すなわち、資金利益、非資金利益、一般貸倒引当金繰入額である。銀行は預金者から預かったお金（預金）を主な資金源として、企業や家計等への貸出、または、債券や株式といった有価証券への投資などにより、資金運用を行っている。こうした資金運用業務の収支が資金利益である。資金利益は、銀行収益全体の大宗を占めていることから、銀行の実力を測る指標の一つとして重視されている。非資金利益は資金利益以外のもので、役務取引等利益、特定取引利益、その他業務利益を含む。長期的に低金利にある場合、今後の収益確保のためには、都市銀行も地方銀行も、非資金利益を引き上げることの重要性をますます認識している。一般貸倒引当金繰入額は融資資金回収の可否を判断する指標として、クラスター分析の特徴量としても選ばれている。 NII、NNI、PROを用いて、資金利益、非資金利益、一般貸倒

"""
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
