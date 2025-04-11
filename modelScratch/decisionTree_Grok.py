import numpy as np
from collections import Counter

class Node:
    def __init__(self, feature=None, threshold=None, left=None, right=None, value=None):
        """
        Node untuk struktur pohon keputusan
        - feature: indeks fitur untuk split
        - threshold: nilai ambang untuk split
        - left: anak kiri (Node)
        - right: anak kanan (Node)
        - value: nilai kelas jika node adalah daun
        """
        self.feature = feature
        self.threshold = threshold
        self.left = left
        self.right = right
        self.value = value

class DecisionTreeClassifier:
    def __init__(self, max_depth=None, min_samples_split=2):
        """
        Inisialisasi Decision Tree Classifier
        - max_depth: kedalaman maksimum pohon
        - min_samples_split: jumlah minimum sampel untuk split
        """
        self.max_depth = max_depth
        self.min_samples_split = min_samples_split
        self.root = None

    def fit(self, X, y):
        """
        Melatih model dengan data X (fitur) dan y (target)
        """
        self.root = self._grow_tree(X, y)
    
    def _grow_tree(self, X, y, depth=0):
        """
        Rekursif membangun pohon keputusan
        """
        n_samples, n_features = X.shape
        n_classes = len(np.unique(y))

        # Kriteria berhenti
        if (self.max_depth is not None and depth >= self.max_depth) or \
           n_samples < self.min_samples_split or \
           n_classes == 1:
            leaf_value = self._most_common_label(y)
            return Node(value=leaf_value)

        # Cari split terbaik
        best_feature, best_threshold = self._best_split(X, y, n_features)

        # Jika tidak ada split yang valid, kembalikan daun
        if best_feature is None:
            leaf_value = self._most_common_label(y)
            return Node(value=leaf_value)

        # Buat split
        left_idxs = X[:, best_feature] <= best_threshold
        right_idxs = ~left_idxs

        # Pastikan ada data untuk kedua cabang
        if sum(left_idxs) == 0 or sum(right_idxs) == 0:
            leaf_value = self._most_common_label(y)
            return Node(value=leaf_value)

        # Bangun anak kiri dan kanan secara rekursif
        left = self._grow_tree(X[left_idxs], y[left_idxs], depth + 1)
        right = self._grow_tree(X[right_idxs], y[right_idxs], depth + 1)

        return Node(best_feature, best_threshold, left, right)

    def _best_split(self, X, y, n_features):
        """
        Mencari fitur dan threshold terbaik untuk split
        """
        best_gain = -1
        best_feature = None
        best_threshold = None

        for feature in range(n_features):
            # Ambil semua nilai unik untuk fitur ini
            thresholds = np.unique(X[:, feature])
            
            for threshold in thresholds:
                # Lakukan split
                left_idxs = X[:, feature] <= threshold
                right_idxs = ~left_idxs

                # Pastikan kedua sisi memiliki data
                if sum(left_idxs) == 0 or sum(right_idxs) == 0:
                    continue

                # Hitung information gain
                gain = self._information_gain(y, left_idxs, right_idxs)

                if gain > best_gain:
                    best_gain = gain
                    best_feature = feature
                    best_threshold = threshold

        return best_feature, best_threshold

    def _information_gain(self, y, left_idxs, right_idxs):
        """
        Menghitung information gain menggunakan Gini impurity
        """
        parent_gini = self._gini_impurity(y)
        
        n = len(y)
        n_left = sum(left_idxs)
        n_right = n - n_left

        if n_left == 0 or n_right == 0:
            return 0

        # Hitung gini impurity untuk anak kiri dan kanan
        child_gini = (n_left / n) * self._gini_impurity(y[left_idxs]) + \
                     (n_right / n) * self._gini_impurity(y[right_idxs])

        return parent_gini - child_gini

    def _gini_impurity(self, y):
        """
        Menghitung Gini impurity untuk array target
        """
        _, counts = np.unique(y, return_counts=True)
        probabilities = counts / len(y)
        return 1 - np.sum(probabilities ** 2)

    def _most_common_label(self, y):
        """
        Mengembalikan label yang paling umum dalam array
        """
        counter = Counter(y)
        return counter.most_common(1)[0][0]

    def predict(self, X):
        """
        Memprediksi kelas untuk data baru
        """
        return np.array([self._traverse_tree(x, self.root) for x in X])

    def _traverse_tree(self, x, node):
        """
        Melintasi pohon untuk satu sampel
        """
        if node.value is not None:
            return node.value

        if x[node.feature] <= node.threshold:
            return self._traverse_tree(x, node.left)
        return self._traverse_tree(x, node.right)

# Contoh penggunaan
if __name__ == "__main__":
    # Contoh data sederhana
    from sklearn.datasets import make_classification
    from sklearn.model_selection import train_test_split
    from sklearn.metrics import accuracy_score

    # Buat data sintetis
    X, y = make_classification(n_samples=100, n_features=4, n_classes=2, random_state=42)
    
    # Split data
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Inisialisasi dan latih model
    clf = DecisionTreeClassifier(max_depth=3, min_samples_split=2)
    clf.fit(X_train, y_train)

    # Prediksi
    y_pred = clf.predict(X_test)

    # Hitung akurasi
    accuracy = accuracy_score(y_test, y_pred)
    print(f"Akurasi: {accuracy:.2f}")