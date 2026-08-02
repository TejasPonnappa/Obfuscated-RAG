import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

class DifferentialPrivacyAgent:
    def __init__(self, epsilon: float = 2.0, sensitivity: float = 1.0):
        self.epsilon = epsilon
        self.sensitivity = sensitivity
        self.scale = self.sensitivity / self.epsilon

    def inject_laplace_noise(self, embeddings: np.ndarray) -> np.ndarray:
        noise = np.random.laplace(loc=0.0, scale=self.scale, size=embeddings.shape)
        return embeddings + noise

class FastDiscriminator:
    def __init__(self):
        self.model = LogisticRegression(max_iter=1000)

    def evaluate_noise(self, clean_vectors: np.ndarray, noisy_vectors: np.ndarray) -> float:
        if clean_vectors.shape[0] < 5:
            clean_vectors = np.tile(clean_vectors, (5, 1))
            noisy_vectors = np.tile(noisy_vectors, (5, 1))
            
        y_clean = np.zeros(clean_vectors.shape[0])
        y_noisy = np.ones(noisy_vectors.shape[0])

        X = np.vstack((clean_vectors, noisy_vectors))
        y = np.concatenate((y_clean, y_noisy))

        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.3, random_state=42, stratify=y
        )
        
        self.model.fit(X_train, y_train)
        predictions = self.model.predict(X_test)
        
        return accuracy_score(y_test, predictions)

def optimize_privacy_budget(clean_vectors: np.ndarray, target_accuracy: float = 0.55):
    """
    The GAN Loop: Pits Blurrer vs Detective.
    Returns the optimal epsilon and the safely obfuscated vectors.
    """
    current_epsilon = 2.0
    detective = FastDiscriminator()
    
    for epoch in range(5): 
        blurrer = DifferentialPrivacyAgent(epsilon=current_epsilon)
        noisy_vectors = blurrer.inject_laplace_noise(clean_vectors)
        
        accuracy = detective.evaluate_noise(clean_vectors, noisy_vectors)
        
        if accuracy <= target_accuracy:
            return current_epsilon, noisy_vectors
        else:
            current_epsilon *= 1.5 
            
    return current_epsilon, noisy_vectors