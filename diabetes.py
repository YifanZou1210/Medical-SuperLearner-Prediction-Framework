import pandas as pd
import numpy as np
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.model_selection import train_test_split, cross_val_predict
from sklearn.decomposition import PCA
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score


# -------------------------
# Step 1: 数据预处理
# -------------------------

def load_data(filepath):
    """
    从 CSV 文件中加载数据
    """
    return pd.read_csv(filepath)

def remove_outliers(df, multiplier=1.5):
    """
    使用 IQR 方法删除异常值 (仅对数值型数据有效)
    Remove outliers from a DataFrame using the IQR method (only works for numeric data)
    参数 / Parameters:
    - df: pd.DataFrame，待处理的数据集 / DataFrame to process.
    - multiplier: float，默认1.5，用于确定IQR范围，即下界 = Q1 - multiplier * IQR，上界 = Q3 + multiplier * IQR。
      Multiplier for the IQR range, default is 1.5. Outlier thresholds are set as:
      lower_bound = Q1 - multiplier * IQR, upper_bound = Q3 + multiplier * IQR.
    返回:
    - pd.DataFrame，移除异常值后的数据集 / DataFrame with outliers removed.
    """
    # 获取所有数值型列的名称，确保后续支队这些列进行IQR异常值处理
    # np.number只包含数据类型属于数值型的数据列，返回值是新的df，只包含选定数据类型的列并通过columns属性提取列的标签
    numeric_cols = df.select_dtypes(include=[np.number]).columns
    print('numeric_cols', numeric_cols)
    # 遍历每个数值型列
    for col in numeric_cols:
        # 分解：对每个列单出处理。先计算第一四分位数(Q1)和第三四分位数(Q3)
        """
        df[col]获取df中名称为col的那一列数据，返回pandas series对象，包含该列所有值
        quantile()对series对象调用计算指定分位数，首先对序列中数值大小排序，找出使25%数据落在其左侧的值
        """
        Q1 = df[col].quantile(0.25)
        Q3 = df[col].quantile(0.75)
        # 然后计算四分位距 (IQR)
        IQR = Q3 - Q1
        # print(Q1,Q3,IQR)
        # 定义下界和上界，通常默认为1.5确定异常值检测的阈值，超过这个范围的值被认为是异常值
        lower_bound = Q1 - multiplier * IQR
        upper_bound = Q3 + multiplier * IQR
        # 条件过滤：筛选出在 [lower_bound, upper_bound] 内的行
        df = df[(df[col] >= lower_bound) & (df[col] <= upper_bound)]
    return df


def impute_missing(df):
    """
    使用均值填充数值型数据缺失值：对DF中的缺失值进行填充，尤其是数值型数据，通过计算各列的均值来替换缺失值NaN
    """
    numeric_cols = df.select_dtypes(include=[np.number]).columns
    # SimpleImpute来自Scikit-learn库，设置mean表示采用均值填充，对于每一列计算该列的均值，并用均值替换缺失的值
    imputer = SimpleImputer(strategy='mean')
    print('imputer', imputer)
    # 此方法先计算每一列的均值（拟合），然后将缺失值替换为相应均值（转换），然后重新赋值给DF来更新数值
    df[numeric_cols] = imputer.fit_transform(df[numeric_cols])
    return df

# 为什么要对数据集进行normalize:当数据波动加大的数值特征比如Age, BMI等直接用于举例计算比如K-means/构建主成分时，数值较高的特征会对计算结果产生更大影响导致算法偏颇
def normalize_data(df):
    """
    对数值型数据进行标准化（均值为0，标准差为1）
    """
    numeric_cols = df.select_dtypes(include=[np.number]).columns
    # 对每个特征列计算均值和标准差，并将数据转换为标准正态分布Z-score
    scaler = StandardScaler()
    # fit计算每个数值型特征的均值和标准差
    # transform将数据按照计算得出的均值和标准差进行标准转换，使得每个特征的均值为0
    df[numeric_cols] = scaler.fit_transform(df[numeric_cols])
    return df


def preprocess_data(filepath):
    """
    加载数据、去除异常值、填补缺失值并归一化
    """
    df = load_data(filepath)
    print(df)
    df = remove_outliers(df)
    df = impute_missing(df)
    df = normalize_data(df)
    return df


# -------------------------
# Step 2: 无监督生成标签
# -------------------------

def generate_labels(df, features, outcome_col='Outcome'):
    """
    使用 KMeans 聚类：
      - 针对给定特征（例如 ['Age', 'Total_Bilirubin', 'Albumin']）聚类成2个簇
      - 根据其中一个关键指标（这里选择 features[1]，即 Total_Bilirubin）计算均值，
        将均值较高的簇标记为 “患病”（Outcome=1），较低的标记为 “无病”（Outcome=0）
    """
    X = df[features]
    kmeans = KMeans(n_clusters=2, random_state=42)
    df['Cluster'] = kmeans.fit_predict(X)
    # 计算聚类后，选定特征（例如 Total_Bilirubin）的均值
    cluster_means = df.groupby('Cluster')[features[1]].mean().to_dict()
    diabetes_cluster = max(cluster_means, key=cluster_means.get)
    df[outcome_col] = df['Cluster'].apply(lambda x: 1 if x == diabetes_cluster else 0)
    df.drop(columns=['Cluster'], inplace=True)
    return df


# -------------------------
# Step 3: 特征提取 —— 训练/测试划分及 PCA 降维
# -------------------------

def split_and_pca(df, outcome_col='Outcome', n_components=3, test_size=0.2, random_state=42):
    """
    将数据集分为训练集和测试集，再使用 PCA 将特征降为 n_components 个主成分.
    注意：PCA 仅对特征进行降维，不包括标签列.
    """
    X = df.drop(columns=[outcome_col])
    y = df[outcome_col]
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state
    )
    pca = PCA(n_components=n_components, random_state=random_state)
    X_train_pca = pca.fit_transform(X_train)
    X_test_pca = pca.transform(X_test)
    return X_train_pca, X_test_pca, y_train, y_test, pca


# -------------------------
# Step 4: 超级学习器 —— Stack 集成
# -------------------------

def train_super_learner(X_train, y_train, cv=5):
    """
    训练超级学习器：
     - 定义三个基础分类器：朴素贝叶斯、神经网络 (MLPClassifier) 和 KNN
     - 利用交叉验证获得每个基础模型的预测概率（正类概率）
     - 拼接结果作为 meta 特征，并用决策树作为元学习器进行训练
    """
    model_nb = GaussianNB()
    model_nn = MLPClassifier(max_iter=500, random_state=42)
    model_knn = KNeighborsClassifier()
    base_models = [model_nb, model_nn, model_knn]

    meta_features = []
    for model in base_models:
        preds = cross_val_predict(model, X_train, y_train, cv=cv, method='predict_proba')[:, 1]
        meta_features.append(preds.reshape(-1, 1))
        # 在整个训练集上拟合基础模型
        model.fit(X_train, y_train)

    meta_X_train = np.hstack(meta_features)

    meta_model = DecisionTreeClassifier(random_state=42)
    meta_model.fit(meta_X_train, y_train)

    return base_models, meta_model


def predict_super_learner(X, base_models, meta_model):
    """
    对输入数据 X ，首先使用所有基础模型获得预测概率，
    拼接为 meta 特征，再由元学习器输出最终分类结果.
    """
    meta_features = []
    for model in base_models:
        preds = model.predict_proba(X)[:, 1]
        meta_features.append(preds.reshape(-1, 1))
    meta_X = np.hstack(meta_features)
    return meta_model.predict(meta_X)


def evaluate_super_learner(X_test, y_test, base_models, meta_model):
    y_pred = predict_super_learner(X_test, base_models, meta_model)
    accuracy = accuracy_score(y_test, y_pred)
    return accuracy


# -------------------------
# Step 5: 主流程
# -------------------------

def main():
    # 数据文件路径，确保 data 文件夹中存在 Liver_disease_data.csv
    data_filepath = 'data/Liver_disease_data.csv'

    # Step 1: 数据预处理得到dataframe
    df = preprocess_data(data_filepath)
    print("数据预处理完成，样本数量：", len(df))

    # Step 2: 生成标签
    features_for_clustering = ['Age',  'LiverFunctionTest', 'BMI']
    df = generate_labels(df, features=features_for_clustering, outcome_col='Outcome')
    print("标签生成完成，Outcome 分布:")
    print(df['Outcome'].value_counts())

    # Step 3: 特征提取 (数据集划分 + PCA降维)
    X_train, X_test, y_train, y_test, pca_model = split_and_pca(df, outcome_col='Outcome', n_components=3)
    print("PCA 降维完成，训练集维度：", X_train.shape, "测试集维度：", X_test.shape)

    # Step 4: 超级学习器训练及评估
    base_models, meta_model = train_super_learner(X_train, y_train, cv=5)
    accuracy = evaluate_super_learner(X_test, y_test, base_models, meta_model)
    print("在测试集上的准确率：{:.2f}%".format(accuracy * 100))


if __name__ == '__main__':
    main()

