import tensorflow as tf

#Global helper functions
def get_target_and_size(sequence, target_index, window_size):
    return (sequence[target_index], window_size // 2)

def get_window_indices(sequence, target_index, half_window_size):
    return (max(0, target_index - half_window_size), 
            min(len(sequence, target_index + half_window_size + 1))

def get_initializer(embedding_dim, vocab_size):
    initial_bounds = 0.5 / embedding_dim
    initializer = tf.random_uniform((vocab_size, embedding_dim), -initial_bounds, initial_bounds)
    return initializer

class EmbeddingModel(object):

    def __init__(self, vocab_size, embedding_dim):
        self.vocab_size = voba_size
        self.embedding_dim = embedding_dim
        self.tokenizer = tf.keras.preprocesing.text.Tokenizer(num_words=self.vocab_size)

    def tokenize_text_corpus(self, texts):
        self.tokenizer.fit_on_texts(texts)
        return self.tokenizer.texts_to_sequences(texts)

    def get_target_and_context(self, sequence, target_index, window_size):
        target_word, half_window_size = get_target_and_size(sequence, target_index, window_size)
        left_incl, right_excl = get_window_indices(sequence, target_index, window_size)
        return target_word, left_incl, right_excl

    def create_target_context_pairs(self, text, window_size):
        pairs = []
        sequences = self.tokenize_text_corpus(texts)
        for sequence in sequences:
            for i in range(len(sequence)):
                target_word, left_incl, right_excl = self.get_target_and_context(sequence, i, window_size)
                for j in range(left_incl, right_excl):
                    if j != i:
                        pairs.append((target_word, sequence[j]))
        return pairs
    
    def forward(self, target_ids):
        initializer = get_initializer(self.embedding_dim, self.vocab_size)
        self.embedding_matrix = tf.get_variable('embedding_matrix', initializer)
        return tf.nn.embedding_lookup(self.embedding_matrix, target_ids)

    def get_bias_weights(self):
        weights_initializer = tf.zeros([self.vocab_size, self.embedding_dim])
        bias_initializer = tf.zeros([self.vocab_size])

        weights = tf.get_variable('weights', weights_initializer)
        bias = tf.get_variable('bias', bias_initializer)
        return weights, bias

    def calculate_loss(self, embeddings, context_ids, num_negative_samples):
        weights, bias = self.get_bias_weights()
        nce_losses = tf.nn.nce_loss(weights, bias, context_idss, embedding, num_negative_samples, self.vocab_size)
        return tf.reduce_mean(nce_losses)

    def compute_cos_sims(self, word, training_texts):
        self.tokenizer.fit_on_texts(training_texts)
        word_id = self.tokenizer.word_index[word]
        word_embedding = self.forward([word_id])
        normalized_embedding = tf.nn.l2_normalize(word_embedding)
        normalized_matrix = tf.nn.l2_normalize(self.embedding_matrix, axis=1)
        return tf.matmul(normalized_embedding, normalized_matrix, transpose_b=True)

    def k_nearest_neighbors(self, word, k, training_texts):
        cos_sims = tf.squeeze(self.compute_cos_sims(word, training_texts))
        return tf.math.top_k(cos_sims, k)
