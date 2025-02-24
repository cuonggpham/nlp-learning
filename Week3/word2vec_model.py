import gensim
import gensim.downloader as api
import numpy as np
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE

# Load a small dataset (text8 corpus from gensim)
dataset = api.load("text8")

# Train Word2Vec model
model = gensim.models.Word2Vec(dataset, vector_size=100, window=5, min_count=5, workers=4)

# Choose a word to visualize similar words
word = "king"
similar_words = [word] + [w for w, _ in model.wv.most_similar(word, topn=5)]

# Get word vectors
word_vectors = np.array([model.wv[w] for w in similar_words])

# Reduce dimensionality using PCA
pca = PCA(n_components=2)
word_vectors_pca = pca.fit_transform(word_vectors)

# Reduce dimensionality using t-SNE
tsne = TSNE(n_components=2, random_state=42)
word_vectors_tsne = tsne.fit_transform(word_vectors)

def plot_words(word_vectors, words, title):
    plt.figure(figsize=(8, 6))
    plt.scatter(word_vectors[:, 0], word_vectors[:, 1], edgecolors='k', c='red')
    for word, (x, y) in zip(words, word_vectors):
        plt.text(x + 0.05, y + 0.05, word, fontsize=12)
    plt.title(title)
    plt.show()

# Plot PCA visualization
plot_words(word_vectors_pca, similar_words, "Word Embeddings Visualization using PCA")

# Plot t-SNE visualization
plot_words(word_vectors_tsne, similar_words, "Word Embeddings Visualization using t-SNE")
