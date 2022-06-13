import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import LabelEncoder
from sklearn import preprocessing
from sklearn.decomposition import PCA
from sklearn.metrics import silhouette_samples, silhouette_score
import matplotlib.cm as cm
from sklearn.cluster import KMeans
# 获取数据
car = pd.read_csv('./car_price.csv')
print(car.head())

# 查看数据类型和非空、重复值
car.info()
print(car.duplicated()) # 检查重复值

# 提取变量特征数据（除"car_ID"和"CarName"）
car_feature = car.drop(['car_ID', 'CarName'], axis=1)
print(car_feature)

# 查看连续数值型情况，并检查是否有异常值
car_feature.describe()
print(car_feature.describe())
# describe() 参考网页：https://blog.csdn.net/sala_lxw/article/details/104500381?spm=1001.2101.3001.6650.1&utm_medium=distribute.pc_relevant.none-task-blog-2%7Edefault%7ECTRLIST%7Edefault-1-104500381-blog-80144660.pc_relevant_blogantidownloadv1&depth_1-utm_source=distribute.pc_relevant.none-task-blog-2%7Edefault%7ECTRLIST%7Edefault-1-104500381-blog-80144660.pc_relevant_blogantidownloadv1&utm_relevant_index=1

# 绘制箱线图
# 提取连续数值型数据的列名
cate_columns = ['symboling', 'fueltype', 'aspiration', 'doornumber', 'carbody', 'drivewheel', 'enginelocation', 'enginetype', 'fuelsystem', 'cylindernumber']
columns_name = car_feature.columns.drop(cate_columns)
print(columns_name)
fig = plt.figure(figsize=(12, 8))
# plt.figure() 参考网页：https://blog.csdn.net/m0_37362454/article/details/81511427
i = 1
for col in columns_name:
    ax = fig.add_subplot(3, 5, i)
    sns.boxplot(data=car_feature[col], ax=ax)
    i = i + 1
    plt.title(col)

plt.subplots_adjust(wspace=0.4, hspace=0.3)
plt.show()

# 去重查看CarName
car['CarName'].drop_duplicates()
print(car['CarName'].drop_duplicates())  # 验证是否object全部改为数值类型


# 提取类变量类名
cate_columns = ['symboling', 'fueltype', 'aspiration', 'doornumber', 'carbody', 'drivewheel', 'enginelocation', 'enginetype', 'fuelsystem', 'cylindernumber']
for i in cate_columns:
    print(i)
    print(set(car[i]))
car['cylindernumber'] = car.cylindernumber.replace({'two': 2, 'six': 6, 'three': 3, 'four': 4, 'five': 5, 'eight': 8, 'twelve': 12})
print(car['cylindernumber'])


# 利用split，拆分汽车品牌
CarBrand = car['CarName'].str.split(expand=True)[0]
print(CarBrand)
# 去重
print(set(CarBrand))
# 修改品牌名称的不规则命名
CarBrand = CarBrand.replace({'porcshce': 'porsche', 'vokswagen': 'volkswagen', ' Nissan': 'nissan', 'maxda': 'mazda', 'vw': 'volkswagen', 'toyouta': 'toyota'})
print(set(CarBrand))
# 将CarBrand放入原数据集中
car['CarBrand'] = CarBrand

# 车长大小范围为141.1-208.1英寸之间，可划分为6类
bins = [min(car_feature.carlength)-0.01, 145.67, 169.29, 181.10, 192.91, 200.79, max(car_feature.carlength)+0.01]
label = ['A00', 'A0', 'A', 'B', 'C', 'D']
carSize = pd.cut(car_feature.carlength, bins, labels=label)  # pandas.cut() 参考网页：https://zhuanlan.zhihu.com/p/393271215
print(carSize)
car['carSize'] = carSize
car_feature['carSize'] = carSize

# 查看数值型特征的相关系数
df_corr = car_feature.corr()
print(df_corr)
# 绘制相关性热力图
mask = np.zeros_like(df_corr)  # numpy.zeros_like 输入为矩阵x 输出为形状和x一致的矩阵，其元素全部为0
mask[np.triu_indices_from(mask)] = True  # numpy.triu() 参考网页：https://blog.csdn.net/ziqingnian/article/details/112334946
plt.figure(figsize=(10, 10))
with sns.axes_style("white"):
    ax = sns.heatmap(df_corr, mask=mask, square=True, annot=True, cmap='bwr')
ax.set_title('df_corr Variables Relation')
plt.show()

# 剔除carlength
features = car_feature.drop(['carlength'], axis=1)
# 将取值具有大小意义的类别变量数据转变为数值型映射
features1 = features.copy()
# 使用sklean中的LabelEncoder对不具实体数值数据编码
carSize1 = LabelEncoder().fit_transform(features1['carSize'])
features1['carSize'] = carSize1
print(features1)
# 对于离别型数据采用One-Hot编码
cate = features1.select_dtypes(include='object').columns
print(cate)
features1 = features1.join(pd.get_dummies(features1[cate])).drop(cate, axis=1)
print(features1.head())

# 对数值型数据进行归一化
features1 = preprocessing.MinMaxScaler().fit_transform(features1)
features1 = pd.DataFrame(features1)
print(features1.head())

# 对数据集进行PCA降维
pca = PCA(n_components=0.9999)
features2 = pca.fit_transform(features1)
# 降维后，每个主要成分的解释方差占比(解释PC携带的信息多少)
ratio = pca.explained_variance_ratio_
# print('各主成分的解释方差占比：', ratio)
# print('降维后有几个成分：', len(ratio))
# 累计解释方差占比
cum_ratio = np.cumsum(ratio)
# print('累计解释方差占比：', cum_ratio)

# 绘制PCA降维后各成分方差占比的直方图和累计方差占比图
plt.figure(figsize=(8, 6))
X = range(1, len(ratio)+1)
Y = ratio
plt.bar(X, Y, edgecolor='black')
plt.plot(X, Y, 'r.-')
plt.plot(X, cum_ratio, 'b.-')
plt.ylabel('explained_variance_ratio')
plt.xlabel('PCA')
plt.show()

# PCA选择降维保留9个主要成分
pca = PCA(n_components=9)
features3 = pca.fit_transform(features1)
print(features3)
# 统计降维后的累计各成分方差占比和（即解释PC携带的信息多少）
print(sum(pca.explained_variance_ratio_))

# 肘方法看k值，簇内离差平方和
# 对每个k值进行聚类并且记下对于的SSE，画出K和SSE的关系图
sse = []
for i in range(1, 40):
    km = KMeans(n_clusters=i, init='k-means++', n_init=10, max_iter=300, random_state=0)
    km.fit(features3)
    sse.append(km.inertia_)
plt.plot(range(1, 40), sse, marker='*')
plt.xlabel('n_clusters')
plt.ylabel('distortions')
plt.title("The Elbow Method")
plt.show()

# 进行K-Means聚类模型
kmeans = KMeans(n_clusters=5, init='k-means++', n_init=10, max_iter=300, random_state=0)
kmeans.fit(features3)
lab = kmeans.predict(features3)
print(lab)
# 绘制聚类后结果的散点图，查看每簇间距离效果
plt.figure(figsize=(8, 8))
plt.scatter(features3[:, 0], features3[:, 1], c=lab)
for ii in np.arange(205):
    plt.text(features3[ii, 0], features3[ii, 1], s=car.car_ID[ii])
plt.xlabel('PC1')
plt.ylabel('PC2')
plt.title('K-Means PCA')
plt.show()
# 绘制聚类后的3d效果图
plt.figure(figsize=(8, 8))
ax = plt.subplot(111, projection='3d')
ax.scatter(features3[:, 0], features3[:, 1], features3[:, 2], c=lab)
# 视角转换，转换后更易看出簇群
ax.view_init(30, 45)
ax.set_xlabel('PC1')
ax.set_ylabel('PC2')
ax.set_zlabel('PC3')
plt.show()

# 绘制轮廓图和3d散点图

for n_clusters in range(2, 9):
    fig = plt.figure(figsize=(12, 6))
    ax1 = fig.add_subplot(121)
    ax2 = fig.add_subplot(122, projection='3d')

    ax1.set_xlim([-0.1, 1])
    ax1.set_ylim([0, len(features3) + (n_clusters + 1) * 10])
    km = KMeans(n_clusters=n_clusters, init='k-means++', n_init=10, max_iter=300, random_state=0)
    y_km = km.fit_predict(features3)
    silhouette_avg = silhouette_score(features3, y_km)
    print('n_cluster=', n_clusters, 'The average silhouette_score is :', silhouette_avg)

    cluster_labels = np.unique(y_km)
    silhouette_vals = silhouette_samples(features3, y_km, metric='euclidean')
    y_ax_lower = 10
    for i in range(n_clusters):
        c_silhouette_vals = silhouette_vals[y_km == i]
        c_silhouette_vals.sort()
        cluster_i = c_silhouette_vals.shape[0]
        y_ax_upper = y_ax_lower + cluster_i
        color = cm.nipy_spectral(float(i) / n_clusters)
        ax1.fill_betweenx(range(y_ax_lower, y_ax_upper), 0, c_silhouette_vals, edgecolor='none', color=color)
        ax1.text(-0.05, y_ax_lower + 0.5 * cluster_i, str(i))
        y_ax_lower = y_ax_upper + 10

    ax1.set_title('The silhouette plot for the various clusters')
    ax1.set_xlabel('The silhouette coefficient values')
    ax1.set_ylabel('Cluster label')

    ax1.axvline(x=silhouette_avg, color='red', linestyle='--')

    ax1.set_yticks([])
    ax1.set_xticks([-0.1, 0, 0.2, 0.4, 0.6, 0.8, 1.0])

    colors = cm.nipy_spectral(y_km.astype(float) / n_clusters)
    ax2.scatter(features3[:, 0], features3[:, 1], features3[:, 2], marker='.', s=30, lw=0, alpha=0.7, c=colors,
                edgecolor='k')

    centers = km.cluster_centers_
    ax2.scatter(centers[:, 0], centers[:, 1], centers[:, 2], marker='o', c='white', alpha=1, s=200, edgecolor='k')

    for i, c in enumerate(centers):
        ax2.scatter(c[0], c[1], c[2], marker='$%d$' % i, alpha=1, s=50, edgecolor='k')

    ax2.set_title("The visualization of the clustered data.")
    ax2.set_xlabel("Feature space for the 1st feature")
    ax2.set_ylabel("Feature space for the 2nd feature")
    ax2.view_init(30, 45)

    plt.suptitle(("Silhouette analysis for KMeans clustering on sample data "
                  "with n_clusters = %d" % n_clusters),
                 fontsize=14, fontweight='bold')
plt.show()

# 调整选择k=6进行聚类
kmeans = KMeans(n_clusters=6, init='k-means++', n_init=10, max_iter=300, random_state=0)
y_pred = kmeans.fit_predict(features3)
print(y_pred)
# 将聚类后的类目放入原特征数据中
car_df_km = car.copy()
car_df_km['km_result'] = y_pred

# 统计聚类后每个集群的车型数
car_df_km.groupby('km_result')['car_ID'].count()
# 统计每个集群各个的车型数
car_df_km.groupby(by=['km_result', 'CarBrand'])['car_ID'].count()
# 统计每个品牌所属各个集群的车型数
car_df_km.groupby(by=['CarBrand', 'km_result'])['km_result'].count()
print(car_df_km)

# 查看特指车名'vokswagen'车型的聚类集群
df = car_df_km.loc[:, ['car_ID', 'CarName', 'CarBrand', 'km_result']]
print(df.loc[df['CarName'].str.contains('vokswagen')])

# 查看特指车名为‘vokswagen’车型的竞品车型（分类为3的所有车型）
df.loc[df['km_result'] == 3]
print(df.loc[df['km_result'] == 3])

# 查看大众volkswagen品牌各集群内的竞品车型
df_volk = df.loc[df['km_result'] > 2].sort_values(by=['km_result', 'CarBrand'])
print(df_volk)

# 提取分类为3的所有车型特征数据
df3 = car_df_km.loc[car_df_km['km_result'] == 3]
print(df3.head())
# 绘制柱状图查看集群3的车型所有特征分布
df3_1 = df3.drop(['car_ID', 'CarName', 'km_result'], axis=1)
fig = plt.figure(figsize=(20, 20))
i = 1
for c in df3_1.columns:
    ax = fig.add_subplot(7, 4, i)
    if df3_1[c].dtypes == 'int' or df3_1[c].dtypes == 'float':
        sns.histplot(df3_1[c], ax=ax)
    else:
        sns.barplot(df3_1[c].value_counts().index, df3_1[c].value_counts(), ax=ax)
    i += 1
    plt.xlabel('')
    plt.title(c)
plt.subplots_adjust(top=1)
plt.show()

# 对不同车型级别、品牌、车身类型等特征进行数据透视
df2 = df3.pivot_table(index=['carSize', 'carbody', 'CarBrand', 'CarName'])
print(df2)

# 提取集群3中所有A级车
df3_A = df3.loc[df3['carSize'] == 'A']
print(df3_A)
# 查看集群3中A级车的类别型变量的分类情况
ate_col = df3_A.select_dtypes(include='object').columns
df4 = df3_A[ate_col]
print(df4)

# 对集群3中的A级车进行数据透视
df5 = df3_A.pivot_table(index=['CarBrand', 'CarName', 'doornumber', 'aspiration', 'drivewheel'])
print(df5)

# 对油耗进行分析
lab = df3_A['CarName']
fig, ax = plt.subplots(figsize=(10, 8))
ax.barh(range(len(lab)), df3_A['highwaympg'], tick_label=lab, color='red')
ax.barh(range(len(lab)), df3_A['citympg'], tick_label=lab, color='blue')
for i, (highway, city) in enumerate(zip(df3_A['highwaympg'], df3_A[ 'citympg'])):
    ax.text(highway, i, highway, ha='right')
    ax.text(city, i, city, ha='right')
plt.legend(('highwaympg', 'citympg'), loc='upper right')
plt.title('miels per gallon')
plt.show()

# 其他6个特征分析
colors = ['yellow', 'blue', 'green', 'red',  'gray', 'tan', 'darkviolet']
col2 = ['symboling', 'wheelbase', 'enginesize', 'horsepower', 'curbweight', 'price']
data = df3_A[col2]

fig = plt.figure(figsize=(8, 8))
i = 1
for c in data.columns:
    ax = fig.add_subplot(3, 2, i)
    plt.barh(range(len(lab)), data[c], tick_label=lab, color=colors)
    for y, x in enumerate(data[c].values):
        plt.text(x, y, "%s"% x)
    i = i+1
    plt.xlabel('')
    plt.title(c)
plt.subplots_adjust(top=1.2, wspace=0.7)
plt.show()

