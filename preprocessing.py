
import numpy as np
import pandas as pd
from sklearn import cluster
import sklearn.preprocessing as preprocess
from sklearn.cluster import KMeans
from sklearn.cluster import DBSCAN

import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

from sklearn.metrics import silhouette_samples
from sklearn.datasets import make_blobs
from matplotlib import cm

from sklearn.model_selection import GridSearchCV

from sklearn.manifold import TSNE


def plotSilhouette(X, y_km):
    cluster_labels = np.unique(y_km)
    n_clusters = cluster_labels.shape[0]
    silhouette_vals = silhouette_samples(X, y_km, metric = 'euclidean')
    y_ax_lower, y_ax_upper = 0, 0
    yticks = []

    for i, c in enumerate(cluster_labels):
        c_silhouette_vals = silhouette_vals[y_km == c]
        c_silhouette_vals.sort()
        y_ax_upper += len(c_silhouette_vals)
        color = cm.jet(i/n_clusters)

        plt.barh(range(y_ax_lower, y_ax_upper), c_silhouette_vals, height=1.0,
                edgecolor='none', color=color)
        yticks.append((y_ax_lower + y_ax_upper)/2)
        y_ax_lower += len(c_silhouette_vals)

    silhoutte_avg = np.mean(silhouette_vals)
    plt.axvline(silhoutte_avg, color = 'red', linestyle='--')
    plt.yticks(yticks, cluster_labels+1)
    plt.ylabel('K')
    plt.xlabel('Number of Silhouette')
    plt.title("Silhouettes")
    plt.show()

# https://mkjjo.github.io/finance/2019/01/09/kmeans_corp.html
# Elbow 찾기
def findElbow(X, range):
    iterscores = []
    ranges = range

    for i in ranges:
        model = KMeans(n_clusters=i)
        model.fit(X)
        iterscores.append(model.inertia_)

    plt.figure(figsize=(16,8))
    plt.plot(ranges, iterscores, 'bx-')
    plt.xlabel('k')
    plt.ylabel('Distortion')
    plt.title('The Elbow Method showing the optimal k')
    plt.show()






df = pd.read_csv("rawdata_novoice.csv")

# print(df.info())
# print("======")

# print(df.describe())
# print("======")

# print(df.isna().sum())
# print("======")

# labelEncoder = preprocess.LabelEncoder()
# age = labelEncoder.fit_transform(df['age'])
# age_encoder = labelEncoder


oneHotEncoder = preprocess.OneHotEncoder()

live = oneHotEncoder.fit_transform(df['live'].to_numpy().reshape(-1,1))
live_encoder = oneHotEncoder


oneHotEncoder = preprocess.OneHotEncoder()
through = oneHotEncoder.fit_transform(df['through'].to_numpy().reshape(-1,1))
through_encoder = oneHotEncoder


labelEncoder = preprocess.LabelEncoder()
sex = labelEncoder.fit_transform(df['sex'])
sex_encoder = labelEncoder


oneHotEncoder = preprocess.OneHotEncoder()
domain =oneHotEncoder.fit_transform(df['domain'].to_numpy().reshape(-1,1))
domain_encoder = oneHotEncoder


oneHotEncoder = preprocess.OneHotEncoder()
transportation = oneHotEncoder.fit_transform(df['transportation'].to_numpy().reshape(-1,1))
transportation_encoder = oneHotEncoder



columns = set([])

most_app = df['most_app']
for i in range(0, len(most_app)):
    items = most_app.iloc[i].split(',')
    for k in range(0, len(items)):
        columns.add(items[k].strip())
columns = list(columns)


# print(tmp.issuperset({'날씨'}))
most_app_onehot = pd.DataFrame({}, columns=columns)
most_app_onehot = most_app_onehot.astype("int64")

for i in range(0, len(most_app)):
    onehottmp = np.zeros([18]).astype("int64")
    items = most_app.iloc[i].split(',')
    for k in range(0, len(items)):
        onehottmp[columns.index(items[k].strip())] = 1
    most_app_onehot.loc[len(most_app_onehot)] = onehottmp


dft = pd.DataFrame({
    "age": df["age"],
    "likes": df["likes"],
    "sex": sex
})

dft = pd.concat([dft, most_app_onehot], axis=1)

# live
tmp = pd.DataFrame(live.toarray(), columns=live_encoder.categories_)
dft = pd.concat([dft, tmp], axis=1)

# through
tmp = pd.DataFrame(through.toarray(), columns=through_encoder.categories_)
dft = pd.concat([dft, tmp], axis=1)

# domain
tmp = pd.DataFrame(domain.toarray(), columns=domain_encoder.categories_)
dft = pd.concat([dft, tmp], axis=1)

# transportation
tmp = pd.DataFrame(transportation.toarray(), columns=transportation_encoder.categories_)
dft = pd.concat([dft, tmp], axis=1)



print(dft)
print(dft.info())


print("========================")
# 원룸
# print(dft[dft.iloc[:,25] == 1])
# dft = dft[dft.iloc[:,25] == 1]
# 가족과 함께
# print(dft[dft.iloc[:,21] == 1])
# dft = dft[dft.iloc[:,21] == 1]
# 학생
# print(dft[dft.iloc[:,37] == 1])
# dft = dft[dft.iloc[:,37] == 1]
# 직장인
# print(dft[dft.iloc[:,35] == 1])
# dft = dft[dft.iloc[:,35] == 1]
# 성별(남)
# print(dft[dft.iloc[:,2] == 0])
# dft = dft[dft.iloc[:,2] == 0]
# 성별(여)
# print(dft[dft.iloc[:,2] == 1])
# dft = dft[dft.iloc[:,2] == 1]
# 자동차X
# print(dft[dft.iloc[:,41] == 0])
# dft = dft[dft.iloc[:,41] == 0]
# 자동차O
print(dft[dft.iloc[:,41] == 1])
dft = dft[dft.iloc[:,41] == 1]


print("========================")

# Elbow 찾기
findElbow(dft, range(1,10))



model = KMeans()

grid = GridSearchCV(estimator=model, cv=10, param_grid={
    "tol": [100, 10, 1, 0.1, 0.01, 0.001,],
    "n_init": [10, 100, 1000,],
    "n_clusters": [2]
}, n_jobs=20)

grid.fit(dft)

print("best score:", grid.best_score_)
print("best param:", grid.best_params_)
print("best estimator:", grid.best_estimator_)

clusters = grid.best_params_['n_clusters']
print("best cluster:", clusters)






model = grid.best_estimator_
model.fit(dft)
# model = DBSCAN(min_samples=clusters)
# model.fit(dft)





# 실루엣 확인
X, y = make_blobs(n_samples=150, n_features=2, centers=3, cluster_std=0.5, shuffle=True, random_state=0)
plotSilhouette(dft, model.fit_predict(dft))


groups = pd.DataFrame({
    "group" : model.fit_predict(dft),
})


# 결과 합치기
result_df = pd.DataFrame(np.hstack((groups, dft)))
# 컬럼명 지정
cols = list(dft.columns)
cols.insert(0,'group')
result_df.columns = cols




# 시각화 (2차원)
transformed = TSNE(n_components=2).fit_transform(dft)
transformed.shape

xs = transformed[:,0]
ys = transformed[:,1]
plt.scatter(xs,ys, c=result_df['group'])  #라벨은 색상으로 분류됨

plt.show()





# 시각화 (3차원)
# https://bcho.tistory.com/1205
# transformed = TSNE(n_components=3).fit_transform(dft)
# transformed.shape

# xs = transformed[:,0]





# ys = transformed[:,1]
# zs = transformed[:,2]

# fig = plt.figure( figsize=(12, 12))
# ax = Axes3D(fig)
# ax.scatter(xs, ys, zs, c=result_df['group'], alpha=0.5)
# ax.set_xlabel('Sepal lenth')
# ax.set_ylabel('Sepal width')
# ax.set_zlabel('Petal length')
# plt.show()










for i in range(0, clusters):
    print("Group", i)
    print(result_df[result_df['group'] == i].describe())
    result_df[result_df['group'] == i].describe().to_csv("Group " + str(i) + ".csv", encoding='euc-kr')
    print("=====================\n\n")
