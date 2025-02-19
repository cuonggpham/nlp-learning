import re
import numpy as np
import matplotlib.pyplot as plt
from collections import defaultdict, Counter


# Tiền xử lý văn bản: chuyển thành chữ thường và loại bỏ dấu câu
def preprocess_text(text):
    text = text.lower()
    text = re.sub(r'[^\w\s]', '', text)  # Loại bỏ dấu câu
    return text.split()  # Tách thành danh sách từ

# Mô hình Trigram
class TrigramModel:
    def __init__(self, corpus):
        # Sử dụng defaultdict để lưu trữ số lần xuất hiện của bigram và trigram
        self.trigram_counts = defaultdict(Counter)
        self.bigram_totals = Counter()
        self.vocab = set()
        self._build_model(corpus)

    def _build_model(self, corpus):
        # """Xây dựng mô hình bằng cách đếm số lần xuất hiện của bigram và trigram."""
        words = preprocess_text(corpus)
        self.vocab.update(words)  # Cập nhật từ vựng
        
        for i in range(len(words) - 2):
            bigram = (words[i], words[i + 1])
            next_word = words[i + 2]
            
            self.trigram_counts[bigram][next_word] += 1  # Đếm số lần xuất hiện của trigram
            self.bigram_totals[bigram] += 1  # Đếm tổng số lần xuất hiện của bigram

    def predict_next_word(self, word1, word2):
        # """Dự đoán từ tiếp theo có xác suất cao nhất dựa trên hai từ trước."""
        bigram = (word1, word2)
        
        if bigram not in self.trigram_counts:
            return None  # Không có dữ liệu cho bigram này
        
        # Tính xác suất cho mỗi từ tiếp theo
        word_probs = {
            word: count / self.bigram_totals[bigram]
            for word, count in self.trigram_counts[bigram].items()
        }
        
        return max(word_probs, key=word_probs.get)  # Từ có xác suất cao nhất

    def plot_word_distribution(self, word1, word2):
        # """Vẽ biểu đồ phân phối xác suất của các từ tiếp theo sau một cặp từ."""
        bigram = (word1, word2)
        
        if bigram not in self.trigram_counts:
            print(f"Không có dữ liệu cho cặp từ '{word1} {word2}'.")
            return
        
        # Tính xác suất của từng từ tiếp theo
        word_probs = {
            word: count / self.bigram_totals[bigram]
            for word, count in self.trigram_counts[bigram].items()
        }
        
        words, probs = zip(*word_probs.items())
        
        plt.figure(figsize=(10, 6))
        plt.bar(words, probs, color='skyblue')
        plt.xlabel("Từ dự đoán")
        plt.ylabel("Xác suất")
        plt.title(f"Phân phối xác suất của từ tiếp theo sau '{word1} {word2}'")
        plt.show()

# Văn bản huấn luyện mẫu
with open("textdata.txt", "r") as file:
    text_corpus = file.read()

# Khởi tạo và huấn luyện mô hình
model = TrigramModel(text_corpus)

# Nhập từ đầu vào từ bàn phím
test_word1 = input("Nhập từ đầu tiên: ")
test_word2 = input("Nhập từ thứ hai: ") 

#predict next word
predicted_word = model.predict_next_word(test_word1, test_word2)
print(f"Từ tiếp theo dự đoán sau '{test_word1} {test_word2}' là: {predicted_word}")

# Hiển thị phân phối xác suất
model.plot_word_distribution(test_word1, test_word2)
