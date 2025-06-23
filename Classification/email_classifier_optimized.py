## 优化版邮件分类器 - 支持参数化特征选择
import re
import os
from jieba import cut
from itertools import chain
from collections import Counter
import numpy as np
from sklearn.naive_bayes import MultinomialNB
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, accuracy_score
import warnings
warnings.filterwarnings('ignore')

class EmailClassifier:
    """优化版邮件分类器，支持参数化特征选择"""
    
    def __init__(self, feature_method='tfidf', top_features=100):
        """
        初始化分类器
        
        Parameters:
        -----------
        feature_method : str, default='tfidf'
            特征提取方法，可选 'tfidf' 或 'frequency'
        top_features : int, default=100
            提取的特征数量
        """
        self.feature_method = feature_method
        self.top_features = top_features
        self.model = MultinomialNB()
        self.tfidf_vectorizer = None
        self.top_words = None
        self.all_words = []
        
    def preprocess_text(self, filename):
        """读取文本并进行预处理"""
        words = []
        with open(filename, 'r', encoding='utf-8') as fr:
            for line in fr:
                line = line.strip()
                # 过滤无效字符
                line = re.sub(r'[.【】0-9、——。，！~\*]', '', line)
                # 使用jieba.cut()方法对文本切词处理
                line = cut(line)
                # 过滤长度为1的词
                line = filter(lambda word: len(word) > 1, line)
                words.extend(line)
        return words
    
    def get_text_content(self, filename):
        """获取文件的原始文本内容（用于TF-IDF）"""
        content = ""
        with open(filename, 'r', encoding='utf-8') as fr:
            for line in fr:
                line = line.strip()
                # 过滤无效字符
                line = re.sub(r'[.【】0-9、——。，！~\*]', '', line)
                content += line + " "
        return content
    
    def jieba_tokenizer(self, text):
        """自定义jieba分词器（用于TF-IDF）"""
        words = cut(text)
        # 过滤长度为1的词
        words = [word for word in words if len(word) > 1]
        return words
    
    def extract_frequency_features(self):
        """提取高频词特征"""
        print("使用高频词特征提取方法...")
        
        # 加载所有邮件文本
        filename_list = ['邮件_files/{}.txt'.format(i) for i in range(151)]
        
        for filename in filename_list:
            self.all_words.append(self.preprocess_text(filename))
        
        # 统计词频
        freq = Counter(chain(*self.all_words))
        self.top_words = [i[0] for i in freq.most_common(self.top_features)]
        
        # 构建特征向量
        vector = []
        for words in self.all_words:
            word_map = list(map(lambda word: words.count(word), self.top_words))
            vector.append(word_map)
        
        return np.array(vector)
    
    def extract_tfidf_features(self):
        """提取TF-IDF特征"""
        print("使用TF-IDF特征提取方法...")
        
        # 加载所有邮件文本内容
        filename_list = ['邮件_files/{}.txt'.format(i) for i in range(151)]
        texts = []
        
        for filename in filename_list:
            content = self.get_text_content(filename)
            texts.append(content)
        
        # 初始化TF-IDF向量化器
        self.tfidf_vectorizer = TfidfVectorizer(
            tokenizer=self.jieba_tokenizer,
            max_features=self.top_features,
            lowercase=False
        )
        
        # 拟合并转换文本
        tfidf_matrix = self.tfidf_vectorizer.fit_transform(texts)
        
        # 获取特征词
        self.top_words = self.tfidf_vectorizer.get_feature_names_out().tolist()
        
        return tfidf_matrix.toarray()
    
    def train(self):
        """训练模型"""
        print(f"开始训练模型，特征提取方法: {self.feature_method}")
        
        # 根据选择的方法提取特征
        if self.feature_method == 'frequency':
            features = self.extract_frequency_features()
        elif self.feature_method == 'tfidf':
            features = self.extract_tfidf_features()
        else:
            raise ValueError("feature_method must be 'frequency' or 'tfidf'")
        
        # 创建标签 (0-126.txt为垃圾邮件标记为1；127-150.txt为普通邮件标记为0)
        labels = np.array([1]*127 + [0]*24)
        
        print(f"特征矩阵形状: {features.shape}")
        print(f"标签分布: 垃圾邮件 {sum(labels)} 个, 普通邮件 {len(labels) - sum(labels)} 个")
        print(f"前10个特征词: {self.top_words[:10]}")
        
        # 训练模型
        self.model.fit(features, labels)
        print("模型训练完成!")
        
        return features, labels
    
    def predict_single(self, filename):
        """对单个邮件进行分类预测"""
        if self.feature_method == 'frequency':
            # 高频词特征预测
            if self.top_words is None:
                raise ValueError("模型尚未训练，请先调用train()方法")
            
            words = self.preprocess_text(filename)
            current_vector = np.array(
                tuple(map(lambda word: words.count(word), self.top_words))
            )
            result = self.model.predict(current_vector.reshape(1, -1))
            
        elif self.feature_method == 'tfidf':
            # TF-IDF特征预测
            if self.tfidf_vectorizer is None:
                raise ValueError("模型尚未训练，请先调用train()方法")
            
            content = self.get_text_content(filename)
            tfidf_vector = self.tfidf_vectorizer.transform([content])
            result = self.model.predict(tfidf_vector)
        
        return '垃圾邮件' if result == 1 else '普通邮件'
    
    def evaluate_model(self, features, labels, test_size=0.2):
        """评估模型性能"""
        # 分割训练集和测试集
        X_train, X_test, y_train, y_test = train_test_split(
            features, labels, test_size=test_size, random_state=42, stratify=labels
        )
        
        # 重新训练模型（使用训练集）
        self.model.fit(X_train, y_train)
        
        # 预测
        y_pred = self.model.predict(X_test)
        
        # 计算准确率
        accuracy = accuracy_score(y_test, y_pred)
        
        print(f"\n=== 模型性能评估 ({self.feature_method}特征) ===")
        print(f"准确率: {accuracy:.4f}")
        print("\n详细分类报告:")
        print(classification_report(y_test, y_pred, target_names=['普通邮件', '垃圾邮件']))
        
        return accuracy

def main():
    """主函数 - 演示两种特征提取方法的对比"""
    
    print("=== 邮件分类器优化版本 - 特征选择方法对比 ===\n")
    
    # 使用高频词特征
    print("=== 高频词特征方法 ===")
    classifier_freq = EmailClassifier(feature_method='frequency', top_features=100)
    features_freq, labels = classifier_freq.train()
    accuracy_freq = classifier_freq.evaluate_model(features_freq, labels)
    
    # 使用TF-IDF特征
    print("\n=== TF-IDF特征方法 ===")
    classifier_tfidf = EmailClassifier(feature_method='tfidf', top_features=100)
    features_tfidf, labels = classifier_tfidf.train()
    accuracy_tfidf = classifier_tfidf.evaluate_model(features_tfidf, labels)
    
    # 对比结果
    print("\n=== 方法对比 ===")
    print(f"高频词特征准确率: {accuracy_freq:.4f}")
    print(f"TF-IDF特征准确率: {accuracy_tfidf:.4f}")
    
    if accuracy_tfidf > accuracy_freq:
        print("TF-IDF特征方法表现更好！")
        best_classifier = classifier_tfidf
    elif accuracy_freq > accuracy_tfidf:
        print("高频词特征方法表现更好！")
        best_classifier = classifier_freq
    else:
        print("两种方法表现相当！")
        best_classifier = classifier_tfidf
    
    # 测试新邮件分类
    test_files = ['151.txt', '152.txt', '153.txt', '154.txt', '155.txt']
    print("\n=== 邮件分类测试 ===")
    for file in test_files:
        filepath = f'邮件_files/{file}'
        if os.path.exists(filepath):
            try:
                result_freq = classifier_freq.predict_single(filepath)
                result_tfidf = classifier_tfidf.predict_single(filepath)
                print(f'{file}:')
                print(f'  高频词方法: {result_freq}')
                print(f'  TF-IDF方法: {result_tfidf}')
                if result_freq != result_tfidf:
                    print(f'  ⚠️ 两种方法结果不一致')
                print()
            except Exception as e:
                print(f'{file}: 分类出错 - {e}')
        else:
            print(f'{file}: 文件不存在')
    
    # 特征词分析
    print("=== 特征词对比分析 ===")
    print(f"\n高频词特征 - 前20个特征词:")
    print(classifier_freq.top_words[:20])
    
    print(f"\nTF-IDF特征 - 前20个特征词:")
    print(classifier_tfidf.top_words[:20])
    
    # 计算特征词重叠度
    freq_set = set(classifier_freq.top_words)
    tfidf_set = set(classifier_tfidf.top_words)
    overlap = freq_set.intersection(tfidf_set)
    overlap_ratio = len(overlap) / len(freq_set.union(tfidf_set))
    
    print(f"\n特征词重叠情况:")
    print(f"重叠词数量: {len(overlap)}")
    print(f"重叠比例: {overlap_ratio:.2%}")

def test_different_parameters():
    """测试不同参数组合的效果"""
    print("\n=== 参数化测试 ===")
    
    params = [
        ('tfidf', 50),
        ('tfidf', 100), 
        ('tfidf', 200),
        ('frequency', 50),
        ('frequency', 100)
    ]
    
    results = []
    
    for method, n_features in params:
        print(f"\n测试: {method}方法, {n_features}个特征")
        classifier = EmailClassifier(feature_method=method, top_features=n_features)
        features, labels = classifier.train()
        accuracy = classifier.evaluate_model(features, labels)
        results.append((method, n_features, accuracy))
    
    print("\n=== 参数对比结果 ===")
    for method, n_features, accuracy in results:
        print(f"{method} ({n_features}特征): {accuracy:.4f}")
    
    # 找出最佳参数组合
    best_result = max(results, key=lambda x: x[2])
    print(f"\n最佳参数组合: {best_result[0]}方法, {best_result[1]}个特征, 准确率: {best_result[2]:.4f}")

if __name__ == "__main__":
    main()
    test_different_parameters() 