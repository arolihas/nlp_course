import tensorflow as tf

def truncate_sequences(sequence, max_length);
    return sequence[:max_length-1], sequence[1:max_length]

def pad_sequences(sequence, max_length):
    padding_amount = max_length - len(sequence)
    padding = [0 for i in range(padding_amount)]
    return sequence[:-1] + padding, sequence[1:] + padding

def get_initializer(embedding_dim, vocab_size):
    initial_bounds = 0.5 / embedding_dim
    initializer = tf.random_uniform((vocab_size, embedding_dim), -initial_bounds, initial_bounds)
    return initializer


class LanguageModel(object):
    def __init__(self, vocab_size, max_length, num_lstm_units, num_lstm_layers):
        self.vocab_size = vocab_size
        self.max_length = max_length
        self.num_lstm_units = num_lstm_units
        self.num_lstm_layers = num_lstm_layers
        self.tokenizer = tf.keras.preprocessing.text.Tokenizer(num_words=vocab_size)

    def get_input_target_sequence(self, sequence):
        if len(sequence) >= self.max_length:
            input_sequence, target_sequence = truncate_sequences(sequence, self.max_length)
        else:
            input_sequence, target_sequence = pad_sequences(sequence, self.max_length)
        return input_sequence, target_sequence

    def get_input_embeddings(self, input_sequences):
        embedding_dim = int(self.vocab_size ** 0.25)
        initializer = get_initializer(embedding_dim, self.vocab_size)
        self.embedding_matrix = tf.get_variable('embedding_matrix', initializer)
        return tf.nn.embedding_lookup(self.embedding_matrix, input_sequences)

    def lstm_cell(self, dropout_keep_prob):
        cell = tf.nn.rnn_cell.LSTMCell(self.num_lstm_units)
        return tf.nn.rnn_cell.DropoutWrapper(cell, output_keep_prob=dropout_keep_prob)

    def stacked_lstm_cells(self, is_training):
        dropout_keep_prob = 0.5 if is_training else 1.0
        cell_list = [self.lstm_cell(dropout_keep_prob) for i in range(self.num_lstm_layers)]
        return tf.nn.rnn_cell.MultiRNNCell(cell_list)

    def run_lstm(self, input_sequences,  is_training):
        cell = self.stacked_lstm_cells(is_training)
        input_embeddings = self.get_input_embeddings(input_sequences)
        binary_sequences = tf.sign(input_sequences)
        sequence_lengths = tf.reduce_sum(binary_sequences, axis=1)
        lstm_outputs = tf.nn.dynamic_rnn(cell, input_embeddings, sequence_length=sequence_lengths, dtype=tf.float32)
        return lstm_outputs[0], binary_sequences

    def calculate_loss(self, lstm_outputs, binary_sequences, output_sequences):
        logits = tf.layers.dense(lstm_outputs, self.voacb_size)
        batch_sequence_loss(tf.nn.sparse_softmax_cross_entropy_loss(output_sequences, logits))
        unpadded_loss = tf.cast(binary_sequences, tf.float32) * batch_sequence_loss
        return tf.reduce_sum(unpadded_loss)

    def get_word_predictions(self, word_preds, binary_sequences, batch_size):
        row_indices = tf.range(batch_size)
        final_indexes = tf.reduce_sum(binary_sequences, axis=1) - 1
        gather_indices = tf.transpose([row_indices, final_indexes])
        return tf.gather_nd(word_preds, gather_indices)

