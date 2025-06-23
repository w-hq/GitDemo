# 基于BERT的情感分类任务
import os
import torch
from transformers import BertTokenizer, BertForSequenceClassification, Trainer, TrainingArguments
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import numpy as np
from sklearn.metrics import accuracy_score, classification_report
import warnings
warnings.filterwarnings('ignore')

class BERTSentimentClassifier:
    """基于BERT的情感分类器"""
    
    def __init__(self, model_name='bert-base-chinese', use_mirror=True):
        """
        初始化BERT情感分类器
        
        Parameters:
        -----------
        model_name : str, default='bert-base-chinese'
            使用的BERT模型名称
        use_mirror : bool, default=True
            是否使用Hugging Face镜像站
        """
        self.model_name = model_name
        
        # 设置Hugging Face镜像站地址
        if use_mirror:
            self.setup_huggingface_mirror()
        
        # 初始化模型和分词器
        self.tokenizer = None
        self.model = None
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        print(f"使用设备: {self.device}")
        print(f"使用模型: {self.model_name}")
    
    def setup_huggingface_mirror(self):
        """设置Hugging Face镜像站地址"""
        # 设置镜像站环境变量
        mirror_url = "https://hf-mirror.com"
        os.environ['HF_ENDPOINT'] = mirror_url
        
        print(f"已设置Hugging Face镜像站: {mirror_url}")
        print("这将加速模型下载速度")
    
    def load_model(self):
        """加载预训练的BERT模型和分词器"""
        try:
            print("正在加载BERT模型和分词器...")
            
            # 加载分词器
            self.tokenizer = AutoTokenizer.from_pretrained(self.model_name)
            
            # 加载模型（用于序列分类，2个类别：正面/负面）
            self.model = AutoModelForSequenceClassification.from_pretrained(
                self.model_name,
                num_labels=2,
                problem_type="single_label_classification"
            )
            
            # 将模型移动到设备上
            self.model.to(self.device)
            
            print("模型和分词器加载成功！")
            
        except Exception as e:
            print(f"模型加载失败: {e}")
            raise
    
    def get_sentences_by_student_id(self, student_id):
        """
        根据学号末尾两位数字获取对应的句子
        
        Parameters:
        -----------
        student_id : str
            学生学号
            
        Returns:
        --------
        tuple: (影评句子, 外卖评价句子)
        """
        # 影评数据
        movie_reviews = {
            0: "这部电影太精彩了，节奏紧凑毫无冷场，完全沉浸其中！",
            1: "剧情设定新颖不落俗套，每个转折都让人惊喜。",
            2: "导演功力深厚，镜头语言非常有张力，每一帧都值得回味。",
            3: "美术、服装、布景细节丰富，完全是视觉盛宴！",
            4: "是近年来最值得一看的国产佳作，强烈推荐！",
            5: "剧情拖沓冗长，中途几次差点睡着。",
            6: "演员表演浮夸，完全无法让人产生代入感。",
            7: "剧情老套，充满套路和硬凹的感动。",
            8: "对白尴尬，像是AI自动生成的剧本。",
            9: "看完只觉得浪费了两个小时，再也不想看第二遍。"
        }
        
        # 外卖评价数据
        delivery_reviews = {
            0: "食物完全凉了，吃起来像隔夜饭，体验极差。",
            1: "汤汁洒得到处都是，包装太随便了。",
            2: "味道非常一般，跟评论区说的完全不一样。",
            3: "分量太少了，照片看着满满的，实际就几口。",
            4: "食材不新鲜，有异味，感觉不太卫生。",
            5: "食物份量十足，性价比超高，吃得很满足！",
            6: "味道超级赞，和店里堂食一样好吃，五星好评！",
            7: "这家店口味稳定，已经回购好几次了，值得信赖！",
            8: "点单备注有按要求做，服务意识很棒。",
            9: "包装环保、整洁美观，整体体验非常好。"
        }
        
        # 获取学号末尾两位数字
        if len(student_id) < 2:
            raise ValueError("学号长度不足，请提供完整学号")
        
        last_digit = int(student_id[-1])  # 末尾第一位
        second_last_digit = int(student_id[-2])  # 末尾第二位
        
        movie_sentence = movie_reviews[last_digit]
        delivery_sentence = delivery_reviews[second_last_digit]
        
        return movie_sentence, delivery_sentence
    
    def predict_sentiment(self, text):
        """
        预测单个文本的情感倾向
        
        Parameters:
        -----------
        text : str
            待分类的文本
            
        Returns:
        --------
        dict: 包含预测结果和置信度的字典
        """
        if self.model is None or self.tokenizer is None:
            raise ValueError("请先调用load_model()加载模型")
        
        # 文本预处理和编码
        inputs = self.tokenizer(
            text,
            return_tensors='pt',
            max_length=512,
            truncation=True,
            padding=True
        )
        
        # 将输入移动到设备上
        inputs = {k: v.to(self.device) for k, v in inputs.items()}
        
        # 进行预测
        self.model.eval()
        with torch.no_grad():
            outputs = self.model(**inputs)
            predictions = torch.nn.functional.softmax(outputs.logits, dim=-1)
            predicted_class = torch.argmax(predictions, dim=-1).item()
            confidence = predictions[0][predicted_class].item()
        
        # 转换为情感标签
        sentiment = "正面" if predicted_class == 1 else "负面"
        
        return {
            'text': text,
            'sentiment': sentiment,
            'confidence': confidence,
            'raw_prediction': predicted_class
        }
    
    def classify_by_student_id(self, student_id):
        """
        根据学号进行情感分类
        
        Parameters:
        -----------
        student_id : str
            学生学号
            
        Returns:
        --------
        dict: 分类结果
        """
        print(f"正在处理学号: {student_id}")
        print(f"学号末尾两位数字: {student_id[-2:]}")
        
        # 获取对应的句子
        movie_sentence, delivery_sentence = self.get_sentences_by_student_id(student_id)
        
        print("\n=== 根据学号选择的句子 ===")
        print(f"影评句子 (末尾数字 {student_id[-1]}): {movie_sentence}")
        print(f"外卖评价 (倒数第二位数字 {student_id[-2]}): {delivery_sentence}")
        
        # 进行情感分类
        movie_result = self.predict_sentiment(movie_sentence)
        delivery_result = self.predict_sentiment(delivery_sentence)
        
        return {
            'student_id': student_id,
            'movie_review': movie_result,
            'delivery_review': delivery_result
        }
    
    def batch_classify(self, student_ids):
        """
        批量处理多个学号的情感分类
        
        Parameters:
        -----------
        student_ids : list
            学号列表
            
        Returns:
        --------
        list: 所有分类结果
        """
        results = []
        for student_id in student_ids:
            try:
                result = self.classify_by_student_id(student_id)
                results.append(result)
            except Exception as e:
                print(f"处理学号 {student_id} 时出错: {e}")
                continue
        
        return results
    
    def print_results(self, results):
        """
        打印分类结果
        
        Parameters:
        -----------
        results : dict or list
            分类结果
        """
        if isinstance(results, dict):
            results = [results]
        
        print("\n" + "="*60)
        print("BERT情感分类结果")
        print("="*60)
        
        for result in results:
            print(f"\n学号: {result['student_id']}")
            print("-" * 40)
            
            # 影评结果
            movie = result['movie_review']
            print(f"影评: {movie['text']}")
            print(f"情感倾向: {movie['sentiment']} (置信度: {movie['confidence']:.4f})")
            
            # 外卖评价结果
            delivery = result['delivery_review']
            print(f"外卖评价: {delivery['text']}")
            print(f"情感倾向: {delivery['sentiment']} (置信度: {delivery['confidence']:.4f})")


def demo_classification():
    """演示情感分类功能"""
    print("=== BERT情感分类任务演示 ===\n")
    
    # 初始化分类器
    classifier = BERTSentimentClassifier()
    
    # 加载模型
    classifier.load_model()
    
    # 示例学号列表（可以根据实际需要修改）
    sample_student_ids = [
        "2021001234",  # 末尾34
        "2021005678",  # 末尾78
        "2021009876",  # 末尾76
        "2021002345",  # 末尾45
        "2021007890"   # 末尾90
    ]
    
    print("\n正在处理示例学号...")
    
    # 处理每个学号
    for student_id in sample_student_ids:
        try:
            result = classifier.classify_by_student_id(student_id)
            classifier.print_results(result)
            print("\n" + "-"*60 + "\n")
        except Exception as e:
            print(f"处理学号 {student_id} 时出错: {e}")
            continue


def interactive_classification():
    """交互式情感分类"""
    print("=== 交互式BERT情感分类 ===\n")
    
    # 初始化分类器
    classifier = BERTSentimentClassifier()
    
    # 加载模型
    classifier.load_model()
    
    while True:
        try:
            student_id = input("\n请输入学号 (输入 'quit' 退出): ").strip()
            
            if student_id.lower() == 'quit':
                print("感谢使用！")
                break
            
            if len(student_id) < 2:
                print("学号长度不足，请输入至少2位数字的学号")
                continue
            
            # 进行分类
            result = classifier.classify_by_student_id(student_id)
            classifier.print_results(result)
            
        except KeyboardInterrupt:
            print("\n\n程序被用户中断")
            break
        except Exception as e:
            print(f"处理过程中出错: {e}")
            continue


if __name__ == "__main__":
    # 可以选择运行演示版本或交互式版本
    print("请选择运行模式:")
    print("1. 演示模式 (使用预设学号)")
    print("2. 交互式模式 (手动输入学号)")
    
    choice = input("请输入选择 (1 或 2): ").strip()
    
    if choice == "1":
        demo_classification()
    elif choice == "2":
        interactive_classification()
    else:
        print("无效选择，运行演示模式")
        demo_classification() 