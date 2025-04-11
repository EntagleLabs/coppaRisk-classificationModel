import numpy as np
from collections import Counter

# Fungsi untuk menghitung entropi dari suatu dataset
def compute_entropy(y):
    """
    Menghitung entropi untuk label y.
    y: array numpy dari label (target)
    """
    counts = np.bincount(y)
    probabilities = counts[np.nonzero(counts)] / len(y)
    entropy = -np.sum(probabilities * np.log2(probabilities))
    return entropy

# Fungsi untuk mendapatkan informasi gain dari suatu split
def information_gain(y, left_indices, right_indices):
    """
    Menghitung information gain dari pembagian dataset menjadi bagian kiri dan kanan.
    y: array numpy label seluruh data
    left_indices: indeks data di cabang kiri
    right_indices: indeks data di cabang kanan
    """
    parent_entropy = compute_entropy(y)
    
    n = len(y)
    n_left = len(left_indices)
    n_right = len(right_indices)
    
    # Jika salah satu cabang kosong, gain = 0
    if n_left == 0 or n_right == 0:
        return 0

    # Menghitung entropi untuk masing-masing cabang
    left_entropy = compute_entropy(y[left_indices])
    right_entropy = compute_entropy(y[right_indices])
    
    # Weighted average dari entropi anak
    weighted_entropy = (n_left / n) * left_entropy + (n_right / n) * right_entropy
    
    # Information gain
    gain = parent_entropy - weighted_entropy
    return gain

# Struktur Node untuk Decision Tree
class Node:
    def __init__(self, feature_index=None, threshold=None, left=None, right=None, *, value=None):
        """
        Jika 'value' tidak None, maka node ini merupakan node daun.
        feature_index: indeks fitur yang digunakan untuk split
        threshold: nilai threshold untuk split
        left: child node sebelah kiri
        right: child node sebelah kanan
        value: nilai label (jika leaf node)
        """
        self.feature_index = feature_index
        self.threshold = threshold
        self.left = left
        self.right = right
        self.value = value  # Hanya diisi jika node adalah daun

    def is_leaf_node(self):
        return self.value is not None

# Kelas Decision Tree
class DecisionTree:
    def __init__(self, max_depth=10, min_samples_split=2):
        """
        max_depth: kedalaman maksimum pohon (untuk menghindari overfitting)
        min_samples_split: jumlah minimal sampel untuk melakukan split
        """
        self.max_depth = max_depth
        self.min_samples_split = min_samples_split
        self.root = None

    # Fungsi untuk memulai pelatihan pohon keputusan
    def fit(self, X, y):
        """
        Melatih decision tree berdasarkan data X dan label y.
        X: array numpy dua dimensi dengan bentuk (n_samples, n_features)
        y: array numpy label dengan panjang n_samples
        """
        self.n_classes = len(np.unique(y))
        self.n_features = X.shape[1]
        self.root = self._grow_tree(X, y)

    # Fungsi rekursif untuk membangun pohon
    def _grow_tree(self, X, y, depth=0):
        n_samples, n_features = X.shape
        num_labels = len(np.unique(y))
        
        # Hentikan rekursi jika semua label sama, mencapai kedalaman maksimum,
        # atau jumlah sampel kurang dari minimum split
        if (depth >= self.max_depth or num_labels == 1 or n_samples < self.min_samples_split):
            leaf_value = self._majority_vote(y)
            return Node(value=leaf_value)
        
        # Mencari split terbaik
        best_feature, best_threshold, best_gain = None, None, -1
        
        for feature_index in range(n_features):
            feature_values = X[:, feature_index]
            # Untuk fitur numerik, kita coba split pada tiap nilai unik (bisa juga nilai rata-rata antar nilai berturut-turut)
            possible_thresholds = np.unique(feature_values)
            for threshold in possible_thresholds:
                # Buat pembagian berdasarkan threshold
                left_indices = np.where(feature_values <= threshold)[0]
                right_indices = np.where(feature_values > threshold)[0]
                # Hitung information gain
                gain = information_gain(y, left_indices, right_indices)
                if gain > best_gain:
                    best_gain = gain
                    best_feature = feature_index
                    best_threshold = threshold
        
        # Jika informasi gain tidak membaik, buat leaf node
        if best_gain == 0:
            leaf_value = self._majority_vote(y)
            return Node(value=leaf_value)
        
        # Melakukan split terbaik
        feature_values = X[:, best_feature]
        left_indices = np.where(feature_values <= best_threshold)[0]
        right_indices = np.where(feature_values > best_threshold)[0]

        # Rekursif ke cabang kiri dan kanan
        left_subtree = self._grow_tree(X[left_indices, :], y[left_indices], depth + 1)
        right_subtree = self._grow_tree(X[right_indices, :], y[right_indices], depth + 1)
        return Node(feature_index=best_feature, threshold=best_threshold, left=left_subtree, right=right_subtree)

    def _majority_vote(self, y):
        """
        Mengembalikan label terbanyak dari array y.
        """
        counter = Counter(y)
        majority_label = counter.most_common(1)[0][0]
        return majority_label

    # Fungsi untuk melakukan prediksi pada dataset baru
    def predict(self, X):
        """
        Mengembalikan prediksi untuk dataset X.
        X: array numpy dua dimensi dengan bentuk (n_samples, n_features)
        """
        return np.array([self._traverse_tree(x, self.root) for x in X])

    def _traverse_tree(self, x, node):
        """
        Menelusuri pohon untuk menentukan label prediksi untuk sampel x.
        """
        if node.is_leaf_node():
            return node.value
        if x[node.feature_index] <= node.threshold:
            return self._traverse_tree(x, node.left)
        else:
            return self._traverse_tree(x, node.right)

# Contoh penggunaan: Membuat dataset sederhana dan melatih decision tree
if __name__ == '__main__':
    # Contoh dataset sederhana: X = fitur, y = label
    # Misalnya, kita membuat dataset biner sederhana dengan 2 fitur
    X = np.array([
        [2.771244718, 1.784783929],
        [1.728571309, 1.169761413],
        [3.678319846, 2.81281357 ],
        [3.961043357, 2.61995032 ],
        [2.999208922, 2.209014212],
        [7.497545867, 3.162953546],
        [9.00220326, 3.339047188 ],
        [7.444542326, 0.476683375],
        [10.12493903, 3.234550982],
        [6.642287351, 3.319983761]
    ])
    # Label biner (0 dan 1)
    y = np.array([0, 0, 0, 0, 0, 1, 1, 1, 1, 1])

    # Membuat dan melatih model decision tree
    tree = DecisionTree(max_depth=3, min_samples_split=2)
    tree.fit(X, y)

    # Prediksi untuk data training
    predictions = tree.predict(X)
    print("Prediksi:", predictions)
    print("Label sebenarnya:", y)

    # Menghitung akurasi
    accuracy = np.sum(predictions == y) / len(y)
    print("Akurasi:", accuracy)