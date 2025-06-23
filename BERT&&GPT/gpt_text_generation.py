# 基于GPT的文本续写任务
import os
import torch
from transformers import GPT2LMHeadModel, GPT2Tokenizer, AutoTokenizer, AutoModelForCausalLM
import warnings
warnings.filterwarnings('ignore')

class GPTTextGenerator:
    """基于GPT的文本续写器"""
    
    def __init__(self, model_name='uer/gpt2-chinese-cluecorpussmall', use_mirror=True):
        """
        初始化GPT文本续写器
        
        Parameters:
        -----------
        model_name : str, default='gpt2-chinese-cluecorpussmall'
            使用的GPT模型名称
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
        """加载预训练的GPT模型和分词器"""
        try:
            print("正在加载GPT模型和分词器...")
            
            self.tokenizer = AutoTokenizer.from_pretrained(self.model_name)
            self.model = AutoModelForCausalLM.from_pretrained(self.model_name)
            
            print(f"成功加载模型: {self.model_name}")
            
            # 设置pad_token
            if self.tokenizer.pad_token is None:
                self.tokenizer.pad_token = self.tokenizer.eos_token
            
            # 将模型移动到设备上
            self.model.to(self.device)
            self.model.eval()
            
            print("模型和分词器加载成功！")
            
        except Exception as e:
            print(f"模型加载失败: {e}")
            raise
    
    def get_prompt_by_student_id(self, student_id):
        """
        根据学号末尾数字获取对应的句子开头
        
        Parameters:
        -----------
        student_id : str
            学生学号
            
        Returns:
        --------
        str: 对应的句子开头
        """
        # 文本续写开头数据
        prompts = {
            0: "如果我拥有一台时间机器",
            1: "当人类第一次踏上火星",
            2: "如果动物会说话，它们最想告诉人类的是",
            3: "有一天，城市突然停电了",
            4: "当我醒来，发现自己变成了一本书",
            5: "假如我能隐身一天，我会",
            6: "我走进了那扇从未打开过的门",
            7: "在一个没有网络的世界里",
            8: "如果世界上只剩下我一个人",
            9: "梦中醒来，一切都变了模样"
        }
        
        # 获取学号末尾数字
        if len(student_id) < 1:
            raise ValueError("学号长度不足，请提供完整学号")
        
        last_digit = int(student_id[-1])  # 末尾第一位
        prompt = prompts[last_digit]
        
        return prompt
    
    def generate_text(self, prompt, max_length=200, num_return_sequences=1, 
                     temperature=0.8, top_k=50, top_p=0.9, do_sample=True):
        """
        根据给定的提示词生成文本
        
        Parameters:
        -----------
        prompt : str
            文本开头/提示词
        max_length : int, default=200
            生成文本的最大长度
        num_return_sequences : int, default=1
            返回的文本序列数量
        temperature : float, default=0.8
            控制生成随机性的温度参数
        top_k : int, default=50
            Top-K采样参数
        top_p : float, default=0.9
            Top-P采样参数
        do_sample : bool, default=True
            是否使用采样
            
        Returns:
        --------
        list: 生成的文本列表
        """
        if self.model is None or self.tokenizer is None:
            raise ValueError("请先调用load_model()加载模型")
        
        # 编码输入文本
        input_ids = self.tokenizer.encode(
            prompt, 
            return_tensors='pt',
            truncation=True,
            max_length=1024
        ).to(self.device)
        
        # 生成文本
        with torch.no_grad():
            outputs = self.model.generate(
                input_ids,
                max_length=max_length,
                num_return_sequences=num_return_sequences,
                temperature=temperature,
                top_k=top_k,
                top_p=top_p,
                do_sample=do_sample,
                pad_token_id=self.tokenizer.eos_token_id,
                eos_token_id=self.tokenizer.eos_token_id,
                repetition_penalty=1.2,  # 防止重复
                no_repeat_ngram_size=3,  # 避免3-gram重复
            )
        
        # 解码生成的文本
        generated_texts = []
        for output in outputs:
            generated_text = self.tokenizer.decode(output, skip_special_tokens=True)
            generated_texts.append(generated_text)
        
        return generated_texts
    
    def generate_by_student_id(self, student_id, max_length=200, num_versions=3):
        """
        根据学号进行文本续写
        
        Parameters:
        -----------
        student_id : str
            学生学号
        max_length : int, default=200
            生成文本的最大长度
        num_versions : int, default=3
            生成版本数量
            
        Returns:
        --------
        dict: 续写结果
        """
        print(f"正在处理学号: {student_id}")
        print(f"学号末尾数字: {student_id[-1]}")
        
        # 获取对应的提示词
        prompt = self.get_prompt_by_student_id(student_id)
        
        print(f"\n=== 根据学号选择的开头句子 ===")
        print(f"开头句子 (末尾数字 {student_id[-1]}): {prompt}")
        
        print(f"\n正在生成 {num_versions} 个版本的续写...")
        
        # 生成多个版本的文本
        generated_texts = self.generate_text(
            prompt, 
            max_length=max_length,
            num_return_sequences=num_versions,
            temperature=0.8,
            top_k=50,
            top_p=0.9
        )
        
        return {
            'student_id': student_id,
            'prompt': prompt,
            'generated_texts': generated_texts,
            'num_versions': len(generated_texts)
        }
    
    def batch_generate(self, student_ids, max_length=200, num_versions=3):
        """
        批量处理多个学号的文本续写
        
        Parameters:
        -----------
        student_ids : list
            学号列表
        max_length : int, default=200
            生成文本的最大长度
        num_versions : int, default=3
            每个学号生成的版本数量
            
        Returns:
        --------
        list: 所有续写结果
        """
        results = []
        for student_id in student_ids:
            try:
                result = self.generate_by_student_id(student_id, max_length, num_versions)
                results.append(result)
            except Exception as e:
                print(f"处理学号 {student_id} 时出错: {e}")
                continue
        
        return results
    
    def print_results(self, results):
        """
        打印文本续写结果
        
        Parameters:
        -----------
        results : dict or list
            续写结果
        """
        if isinstance(results, dict):
            results = [results]
        
        print("\n" + "="*60)
        print("GPT文本续写结果")
        print("="*60)
        
        for result in results:
            print(f"\n学号: {result['student_id']}")
            print("-" * 40)
            print(f"开头句子: {result['prompt']}")
            print(f"生成版本数: {result['num_versions']}")
            
            for i, text in enumerate(result['generated_texts'], 1):
                print(f"\n--- 版本 {i} ---")
                print(text)
                print()
    
    def save_results(self, results, filename="gpt_generation_results.txt"):
        """
        保存续写结果到文件
        
        Parameters:
        -----------
        results : dict or list
            续写结果
        filename : str
            保存的文件名
        """
        if isinstance(results, dict):
            results = [results]
        
        with open(filename, 'w', encoding='utf-8') as f:
            f.write("GPT文本续写结果\n")
            f.write("="*60 + "\n\n")
            
            for result in results:
                f.write(f"学号: {result['student_id']}\n")
                f.write("-" * 40 + "\n")
                f.write(f"开头句子: {result['prompt']}\n")
                f.write(f"生成版本数: {result['num_versions']}\n\n")
                
                for i, text in enumerate(result['generated_texts'], 1):
                    f.write(f"--- 版本 {i} ---\n")
                    f.write(text + "\n\n")
                
                f.write("\n" + "="*60 + "\n\n")
        
        print(f"结果已保存到: {filename}")


def demo_generation():
    """演示文本续写功能"""
    print("=== GPT文本续写任务演示 ===\n")
    
    # 初始化文本生成器
    generator = GPTTextGenerator()
    
    # 加载模型
    generator.load_model()
    
    # 示例学号列表（可以根据实际需要修改）
    sample_student_ids = [
        "2021001230",  # 末尾0
        "2021001231",  # 末尾1
        "2021001232",  # 末尾2
        "2021001233",  # 末尾3
        "2021001234",  # 末尾4
    ]
    
    print("\n正在处理示例学号...")
    
    # 处理每个学号
    all_results = []
    for student_id in sample_student_ids:
        try:
            result = generator.generate_by_student_id(student_id, max_length=150, num_versions=2)
            all_results.append(result)
            
            # 打印单个结果
            generator.print_results(result)
            print("\n" + "-"*60 + "\n")
        except Exception as e:
            print(f"处理学号 {student_id} 时出错: {e}")
            continue
    
    # 保存所有结果
    if all_results:
        generator.save_results(all_results, "demo_generation_results.txt")


def interactive_generation():
    """交互式文本续写"""
    print("=== 交互式GPT文本续写 ===\n")
    
    # 初始化文本生成器
    generator = GPTTextGenerator()
    
    # 加载模型
    generator.load_model()
    
    while True:
        try:
            student_id = input("\n请输入学号 (输入 'quit' 退出): ").strip()
            
            if student_id.lower() == 'quit':
                print("感谢使用！")
                break
            
            if len(student_id) < 1:
                print("学号长度不足，请输入至少1位数字的学号")
                continue
            
            # 获取生成参数
            try:
                max_length = int(input("请输入最大生成长度 (默认200): ").strip() or "200")
                num_versions = int(input("请输入生成版本数 (默认3): ").strip() or "3")
            except ValueError:
                print("使用默认参数: 最大长度200，生成3个版本")
                max_length = 200
                num_versions = 3
            
            # 进行文本续写
            result = generator.generate_by_student_id(student_id, max_length, num_versions)
            generator.print_results(result)
            
            # 询问是否保存结果
            save_choice = input("\n是否保存结果到文件? (y/n): ").strip().lower()
            if save_choice == 'y':
                filename = f"generation_result_{student_id}.txt"
                generator.save_results(result, filename)
            
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
        demo_generation()
    elif choice == "2":
        interactive_generation()
    else:
        print("无效选择，运行演示模式")
        demo_generation() 