import pandas as pd
import numpy as np
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.model_selection import train_test_split, cross_val_predict,GridSearchCV,StratifiedKFold
from sklearn.decomposition import PCA
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score


# -------------------------
# Step 1: 数据预处理阶段
# -------------------------

def load_data(filepath):
    """
    从 CSV 文件中加载数据
    输入：
        filepath: CSV 文件路径
    返回：
        DataFrame 格式的数据
    """
    # 读取 CSV 文件，返回数据集
    return pd.read_csv(filepath)


def remove_outliers(df, multiplier=1.5):
    """
    使用 IQR 方法删除异常值 (仅对数值型数据有效)
    参数:
        df: 待处理的 DataFrame
        multiplier: IQR 倍数，默认值为 1.5，用于确定上下界
    返回:
        移除异常值后的 DataFrame
    """
    # 获取所有数值型列的名称
    # np.number：只包含数值型数据类型的列
    numeric_cols = df.select_dtypes(include=[np.number]).columns
    # 遍历每个数值型列进行异常值处理
    for col in numeric_cols:
        # 计算第一四分位数 (Q1)，即 25% 分位数
        Q1 = df[col].quantile(0.25)
        # 计算第三四分位数 (Q3)，即 75% 分位数
        Q3 = df[col].quantile(0.75)
        # 计算四分位距 IQR = Q3 - Q1
        IQR = Q3 - Q1
        # 根据 IQR 计算异常值上下界
        lower_bound = Q1 - multiplier * IQR
        upper_bound = Q3 + multiplier * IQR
        # 仅保留满足条件的数据行，即在 [lower_bound, upper_bound] 区间内的数据
        df = df[(df[col] >= lower_bound) & (df[col] <= upper_bound)]
    # 返回过滤后的 DataFrame
    return df


def impute_missing(df):
    """
    使用均值填充数值型数据缺失值
    参数:
        df: 包含缺失值的 DataFrame
    返回:
        填充缺失值后的 DataFrame
    """
    # 筛选出所有数值型列
    numeric_cols = df.select_dtypes(include=[np.number]).columns
    # 创建一个 SimpleImputer 对象，策略设置为 'mean' （均值填充）
    imputer = SimpleImputer(strategy='mean')
    # 对数值型列进行 fit_transform 操作：
    # 首先计算每列的均值，然后将缺失值替换为对应的均值
    df[numeric_cols] = imputer.fit_transform(df[numeric_cols])
    return df


def normalize_data(df):
    """
    对数值型数据进行标准化（均值为 0，标准差为 1）
    参数:
        df: 待标准化处理的 DataFrame
    返回:
        标准化后的 DataFrame，数值型特征均转换为标准正态分布数据
    详细说明:
        - 对每个数值型特征计算均值和标准差
        - 使用公式： z = (x - μ) / σ 将数据转换后均值变为 0，标准差变为 1
    """
    # 筛选出所有数值型列
    numeric_cols = df.select_dtypes(include=[np.number]).columns
    # 创建 StandardScaler 对象
    scaler = StandardScaler()
    # 对选中的数值型列执行标准化操作 (fit_transform 计算均值、标准差并转换数据)
    df[numeric_cols] = scaler.fit_transform(df[numeric_cols])
    return df


def preprocess_data(filepath):
    """
    加载数据、去除异常值、填补缺失值并标准化
    参数:
        filepath: 数据文件路径
    返回:
        预处理后的 DataFrame
    流程:
        1. 加载数据
        2. 删除异常值
        3. 填充缺失值
        4. 标准化数据
    """
    df = load_data(filepath)
    df = remove_outliers(df)  # 删除异常值
    df = impute_missing(df)  # 填补缺失值
    df = normalize_data(df)  # 标准化数据，使每个特征均值为0，标准差为1
    return df


# -------------------------
# Step 2: 用于生成标签的无监督学习
# -------------------------

def generate_labels(df, features, outcome_col='Outcome'):
    """
    使用 KMeans 聚类生成标签：
      - 针对三个特征（如 'Glucose', 'BMI', 'Age'）聚类成两个簇
      - 计算每个簇中 'Glucose' 特征的均值，将血糖均值较高的簇标记为“糖尿病”（Outcome=1），
        较低的簇标记为“无糖尿病”（Outcome=0）
      - 将生成的二值标签添加为 Outcome 列
    参数:
        df: 预处理后的 DataFrame
        features: 用于聚类的关键特征列表，如 ['Glucose', 'BMI', 'Age']
        outcome_col: Outcome 列名称，默认 'Outcome'
    返回:
        添加 Outcome 标签后的 DataFrame
    """
    # 提取用于聚类的特征数据（例如血糖、BMI 和年龄）
    X = df[features]
    # 初始化 KMeans 聚类模型，聚类数设为 2，并设置随机种子以保证结果可重复
    kmeans = KMeans(n_clusters=2, random_state=15)
    # 对数据进行聚类，并将每个样本所属的簇标签存入暂时的 'Cluster' 列
    df['Cluster'] = kmeans.fit_predict(X)
    # 针对每个簇计算 'Glucose' 特征的均值，转换为字典形式，键为簇号，值为该簇血糖的均值
    cluster_means = df.groupby('Cluster')['Glucose'].mean().to_dict()
    # 找出血糖均值较高的簇，该簇视为“糖尿病”，返回该簇号
    diabetes_cluster = max(cluster_means, key=cluster_means.get)
    # 根据聚类标签赋予 Outcome 值：属于高血糖均值簇的样本标为 1（糖尿病），其它标为 0（无糖尿病）
    df[outcome_col] = df['Cluster'].apply(lambda x: 1 if x == diabetes_cluster else 0)
    # 删除临时生成的 Cluster 列
    df.drop(columns=['Cluster'], inplace=True)
    # 返回添加 Outcome 标签后的 DataFrame
    return df

# -------------------------
# Step 3: 特征提取 —— 训练/测试划分及 PCA 降维
# -------------------------

def split_and_pca(df, outcome_col='Outcome', n_components=3, test_size=0.2, random_state=42):
    """
    将数据集分为训练集和测试集，再使用 PCA 将特征降为 n_components 个主成分
    参数:
        df: 已添加 Outcome 标签的 DataFrame
        outcome_col: Outcome 列名称（标签列），默认 'Outcome'
        n_components: PCA 降维后的主成分数目，默认 3
        test_size: 测试集比例，默认 20%
        random_state: 随机种子，确保划分结果可重复
    返回:
        X_train_pca: 训练集经过 PCA 降维后的特征矩阵
        X_test_pca: 测试集经过 PCA 转换后的特征矩阵
        y_train: 训练集标签（转换为整型，确保为离散的 0/1）
        y_test: 测试集标签（转换为整型，确保为离散的 0/1）
        pca: 拟合后的 PCA 模型（可用于后续数据转换）
    流程:
        1. 从 DataFrame 中分离特征 X 和标签 y（Outcome 列）
        2. 直接将 y 转换为整型（0/1），而不进行不必要的比较操作
        3. 使用 train_test_split 划分训练集和测试集
        4. 初始化 PCA 对象，并对训练集进行 fit_transform 操作，然后对测试集进行 transform
    """
    # 1. 分离特征和标签：删除 outcome 列得到 X；将 outcome 列赋给 y
    X = df.drop(columns=[outcome_col])
    y = df[outcome_col]

    # 2. 直接将标签转换为整型，确保 y 为离散的 0/1，而非所有样本均为 1
    y = y.astype(int)

    # 3. 使用 train_test_split 划分训练集和测试集
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state
    )

    # 4. 初始化 PCA 对象，设置降维后的主成分数
    pca = PCA(n_components=n_components, random_state=random_state)
    # 在训练集上拟合 PCA 并转换
    X_train_pca = pca.fit_transform(X_train)
    # 使用已拟合的 PCA 对测试集进行转换
    X_test_pca = pca.transform(X_test)

    # 返回降维后的特征矩阵、对应标签及 PCA 模型
    return X_train_pca, X_test_pca, y_train, y_test, pca


# -------------------------
# Step 4: 使用超级学习器进行分类（Stacking）
# -------------------------
def train_super_learner(X_train, y_train, cv=5):
    """
    训练超级学习器：
      - 定义三个基础分类器：朴素贝叶斯、MLP神经网络和KNN，并利用 GridSearchCV 对超参数进行调优
      - 利用分层交叉验证（StratifiedKFold）获得每个基础模型在训练集上的预测概率（正类概率），构造元特征矩阵
      - 使用决策树作为元学习器，对元特征进行训练并调优，获得最佳元模型
    参数:
        X_train: 训练集特征数据（降维后的数据）
        y_train: 训练集标签
        cv: 交叉验证折数，默认 5
    返回:
        base_models: 调优并拟合好的基础模型列表
        best_meta: 调优后的元学习器模型（决策树）
    """
    skf = StratifiedKFold(n_splits=cv, shuffle=True, random_state=42)

    # -------------------------------
    # 定义基础分类器1：GaussianNB（朴素贝叶斯）
    # -------------------------------
    model_nb = GaussianNB()  # 初始化朴素贝叶斯模型
    nb_param_grid = {'var_smoothing': np.logspace(-9, -1, 9)}  # 定义超参数搜索空间
    grid_nb = GridSearchCV(model_nb, nb_param_grid, cv=skf, scoring='accuracy')
    grid_nb.fit(X_train, y_train)  # 在训练集上进行调优拟合
    best_nb = grid_nb.best_estimator_  # 提取最佳模型

    # -------------------------------
    # 定义基础分类器2：MLPClassifier（神经网络）
    # -------------------------------
    model_nn = MLPClassifier(
        random_state=15,
        early_stopping=True,  # 启用早停法
        validation_fraction=0.1,  # 使用10%的训练数据作为验证集
        learning_rate_init=0.0001  # 降低学习率
    )  # 初始化MLP神经网络模型
    nn_param_grid = {
        'hidden_layer_sizes': [(50,), (100,)],  # 隐藏层结构候选
        'max_iter': [1000, 1500, 2000],  # 增加最大迭代次数候选
        'alpha': [1e-5, 1e-4, 1e-3],  # 调整正则化参数候选
        'tol': [1e-4, 1e-5]  # 调整收敛容忍度
    }  # 定义超参数网格
    grid_nn = GridSearchCV(model_nn, nn_param_grid, cv=skf, scoring='accuracy')
    grid_nn.fit(X_train, y_train)  # 调优拟合
    best_nn = grid_nn.best_estimator_  # 选出最佳模型

    # -------------------------------
    # 定义基础分类器3：KNeighborsClassifier（KNN）
    # -------------------------------
    model_knn = KNeighborsClassifier()  # 初始化KNN模型
    knn_param_grid = {'n_neighbors': [3, 5, 7, 9]}  # 定义超参数网格
    grid_knn = GridSearchCV(model_knn, knn_param_grid, cv=skf, scoring='accuracy')
    grid_knn.fit(X_train, y_train)  # 调优拟合
    best_knn = grid_knn.best_estimator_  # 提取最佳模型

    # 将调优后的三个基础模型放入列表中
    base_models = [best_nb, best_nn, best_knn]

    # -------------------------------
    # 构造元特征：利用分层交叉验证获得各基础模型在训练集上的正类预测概率
    # -------------------------------
    meta_features = []  # 初始化存储元特征的列表
    for model in base_models:
        # 使用 cross_val_predict 获得每个模型在分层交叉验证中预测的概率，method='predict_proba'
        preds = cross_val_predict(model, X_train, y_train, cv=skf, method='predict_proba')
        # 判断预测输出的列数
        if preds.shape[1] == 1:
            # 如果只有一列，说明该折可能只有单一类被预测，则补充一列：另一列为 1 - 已有概率
            preds = np.hstack((1 - preds, preds))
        # 取正类概率，即预测数组的第二列
        preds = preds[:, 1]
        # 重塑为列向量（二维数组）后添加到元特征列表中
        meta_features.append(preds.reshape(-1, 1))
        # 使用全部训练数据对该基础模型进行拟合（为后续预测做准备）
        model.fit(X_train, y_train)

    # 横向堆叠所有基础模型的预测概率，形成元特征矩阵
    meta_X_train = np.hstack(meta_features)

    # -------------------------------
    # 定义元学习器：决策树，并进行超参数调优
    # -------------------------------
    meta_model = DecisionTreeClassifier(random_state=15)  # 初始化决策树元模型
    meta_param_grid = {'max_depth': [None, 3, 5, 7], 'min_samples_split': [2, 5, 10]}  # 定义超参数网格
    grid_meta = GridSearchCV(meta_model, meta_param_grid, cv=skf, scoring='accuracy')
    grid_meta.fit(meta_X_train, y_train)  # 在元特征上进行调优拟合
    best_meta = grid_meta.best_estimator_  # 提取最佳决策树元模型

    # 返回调优并拟合好的基础模型列表和最佳元模型
    return base_models, best_meta


def predict_super_learner(X, base_models, meta_model):
    """
    使用训练好的基础模型和元模型对新的数据进行预测
    参数:
        X: 输入特征数据（降维后）
        base_models: 已训练好的基础分类器列表
        meta_model: 已训练好的元学习器（决策树）
    流程:
        1. 对于每个基础模型使用 predict_proba 方法预测正类概率
        2. 将每个模型的预测概率重塑为列向量并横向堆叠形成元特征矩阵
        3. 使用元模型对元特征进行预测，返回最终预测标签
    返回:
        最终预测的标签数组
    """
    meta_features = []  # 初始化存储元特征的列表
    for model in base_models:
        # 使用 predict_proba 获取预测概率，若只有一列则手动补充
        preds = model.predict_proba(X)
        if preds.shape[1] == 1:
            preds = np.hstack((1 - preds, preds))
        # 提取正类概率（第二列）
        preds = preds[:, 1]
        # 将正类概率重塑为二维列向量后加入列表
        meta_features.append(preds.reshape(-1, 1))
    # 将所有基础模型的预测概率拼接为一个元特征矩阵
    meta_X = np.hstack(meta_features)
    # 使用元模型对元特征进行预测，并返回最终分类结果
    return meta_model.predict(meta_X)


def evaluate_super_learner(X_test, y_test, base_models, meta_model):
    """
    在测试集上评估超级学习器的准确率
    参数:
        X_test: 测试集特征数据（降维后）
        y_test: 测试集标签
        base_models: 已训练好的基础分类器列表
        meta_model: 已训练好的元学习器（决策树）
    返回:
        测试集上的分类准确率
    """
    # 调用 predict_super_learner 得到测试集的预测标签
    y_pred = predict_super_learner(X_test, base_models, meta_model)
    # 计算预测标签与真实标签之间的准确率
    accuracy = accuracy_score(y_test, y_pred)
    # 返回测试集准确率
    return accuracy


# -------------------------
# Step 5: 在其他数据集上使用该模型 (liver_disease_data.csv)
# -------------------------

def preprocess_new_data(filepath, outcome_is_last=True):
    """
    对新数据集进行预处理：加载数据、删除异常值、填补缺失值并标准化。
    如果 outcome_is_last 为 True，则将最后一列作为 Outcome 标签，并转换为二元（0/1）格式。
    参数:
        filepath: 新数据集的文件路径
        outcome_is_last: 布尔值，是否将最后一列作为 Outcome 标签，默认 True
    返回:
        预处理后的 DataFrame，其中 Outcome 列为 0/1 类型
    """
    # 加载数据并预处理（去除异常值、填补缺失值、标准化）
    df = preprocess_data(filepath)
    if outcome_is_last:
        # 将 DataFrame 的最后一列作为 Outcome 标签
        outcome_col = df.columns[-1]
        # 如果 Outcome 列的数据类型为 object（例如 'Yes', 'No'），则使用映射转换为数字
        if df[outcome_col].dtype == 'O':
            df[outcome_col] = df[outcome_col].map({'No': 0, 'Yes': 1})
        # 如果 Outcome 列的数据类型为数值型
        elif np.issubdtype(df[outcome_col].dtype, np.number):
            # 提取 Outcome 列中所有唯一值
            unique_vals = np.unique(df[outcome_col])
            # 如果唯一值数量刚好为 2，则将数据转换为整数类型
            if len(unique_vals) == 2:
                df[outcome_col] = df[outcome_col].astype(int)
            # 如果唯一值数量大于2（说明是连续数据），则使用中位数作为阈值将数据二值化
            elif len(unique_vals) > 2:
                median_val = df[outcome_col].median()
                # 大于中位数的设为 1，小于或等于中位数的设为 0
                df[outcome_col] = (df[outcome_col] > median_val).astype(int)
    # 返回预处理后的 DataFrame
    return df




# -------------------------
# 主流程：整合所有步骤
# -------------------------

def main():
    # 使用 diabetes_project.csv 作为训练数据（步骤1-4）
    diabetes_file = 'data/diabetes_project.csv'
    df_train = preprocess_data(diabetes_file)
    print("对 diabetes_project.csv 进行预处理完成，样本数量：", len(df_train))

    # Step 2: 生成标签，选取关键特征：Glucose, BMI and Age
    features_for_clustering = ['Age', 'Glucose', 'BMI']
    df_train = generate_labels(df_train, features=features_for_clustering, outcome_col='Outcome')
    print("生成标签完成，Outcome 分布：")
    print(df_train['Outcome'].value_counts())

    # Step 3: 分割数据并使用 PCA 降维
    X_train, X_test, y_train, y_test, pca_model = split_and_pca(df_train, outcome_col='Outcome', n_components=3)
    print("对 diabetes_project.csv 进行 PCA 降维完成，X_train 维度：", X_train.shape, "X_test 维度：", X_test.shape)

    # Step 4: 使用超级学习器训练分类模型
    base_models, meta_model = train_super_learner(X_train, y_train, cv=5)
    accuracy_train = evaluate_super_learner(X_test, y_test, base_models, meta_model)
    print("在 diabetes_project.csv 测试集上的准确率：{:.2f}%".format(accuracy_train * 100))

    # Step 5: 在其他数据集上使用该模型： liver_disease_data.csv
    liver_file = 'data/Liver_disease_data.csv'
    df_new = preprocess_new_data(liver_file, outcome_is_last=True)
    print("加载并预处理 liver_disease_data.csv 完成，样本数量：", len(df_new))

    outcome_col_new = df_new.columns[-1]
    print("新数据集 Outcome 标签列：", outcome_col_new)

    # 分离特征和标签
    X_new = df_new.drop(columns=[outcome_col_new])
    y_new = df_new[outcome_col_new]
    # 划分新数据集为训练集和测试集，默认 20% 用于测试
    X_train_new, X_test_new, y_train_new, y_test_new = train_test_split(
        X_new, y_new, test_size=0.2, random_state=42
    )

    # 对新数据集进行 PCA 降维（保持主成分数目一致）
    pca_new = PCA(n_components=3, random_state=42)
    X_train_new_pca = pca_new.fit_transform(X_train_new)
    X_test_new_pca = pca_new.transform(X_test_new)

    base_models_new, meta_model_new = train_super_learner(X_train_new_pca, y_train_new, cv=5)
    accuracy_new = evaluate_super_learner(X_test_new_pca, y_test_new, base_models_new, meta_model_new)
    print("在 liver_disease_data.csv 测试集上的准确率：{:.2f}%".format(accuracy_new * 100))


if __name__ == '__main__':
    main()


