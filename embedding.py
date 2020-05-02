import tensorflow as tf

#Global helper functions
def get_target_and_size(sequence, target_index, window_size):
    return (sequence[target_index], window_size // 2)

def get_window_indices(sequence, target_index, half_window_size):
    return (max(0, target_index - half_window_size), 
            min(len(sequence, target_index + half_window_size + 1))

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

