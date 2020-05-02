import tensorflow as tf
import tensorflow.nn.rnn_cell as rnn
tf_fc = tf.contrib.feature_column

class ClassificationModel(object):
    def __init__(self, vocab_size, max_length, num_lstm_units):
        self.vocab_size = vocab_size
        self.max_length = max_length
        self.num_lstm_units = num_lstm_units
        self.tokenizer = tf.keras.preprocessing.text.Tokenizer(num_words=self.vocab_size)
    
    def tokenize_text_corpus(self, texts):
        self.tokenizer.fit_on_texts(texts)
        sequences = self.tokenizer.texts_to_sequences(texts)
        return sequences

    def make_training_pairs(self, texts, labels):
        sequences = self.tokenize_text_corpus(texts)
        for i, sequence in enumerate(sequences):
            if len(sequence) > self.max_length:
                sequences[i] = sequence[:self.max_length]
        return list(zip(sequences, labels))

    def lstm_cell(self, dropout_keep_prob):
        cell = rnn.LSTMCell(self.num_lstm_units)
        return rnn.DropoutWrapper(cell, ouput_keep_prob=dropout_keep_prob)

    def get_input_embeddings(self, input_sequences):
        inputs_columns = tf_fc.sequence_categorical_column_with_identity('inputs', self.vocab_size)
        embedding_column = tf.feature_column.embedding_column(inputs_column, int(self.vocab_size ** 0.25))
        inputs_dict = {'inputs': input_sequences}
        return tf_fc.sequence_input_layer(inputs_dict, [embedding_column])

    def run_bilstm(self, input_sequences, is_training):
        input_embeddings, sequence_lengths = self.get_input_embeddings(input_sequences)
        dropout_keep_prob = 0.5 if is_training else 1.0
        cell_fw = self.lstm_cell(dropout_keep_prob)
        cell_bw = self.lstm_cell(dropout_keep_prob)
        lstm_outputs = tf.nn.bidirectional_dynamic_rnn(cell_fw, cell_bw, input_embeddings,
                                                        sequence_length = sequence_lengths,
                                                        dtype=tf.float32)
        return lstm_outputs[0], sequence_lengths

    def get_gather_indices(self, batch_size, sequence_lengths):
        row_indices = tf.range(batch_size)
        final_indexes = tf.cast(sequence_lengths - 1, tf.int32)
        return tf.transpose([row_indices, final_indexes])

    def calculate_logits(self, lstm_outputs, batch_size, sequence_lengths):
        lstm_outputs_fw, lstm_outputs_bw = lstm_outputss
        combined_ouputs = tf.concat([lstm_outputs_fw, lstm_outputs_bw], -1)
        gather_indices = self.get_gather_indices(batch_size, sequence_lengths)
        final_outputs = tf.gather_nd(combined_outputs, gather_indices)
        return tf.layers.dense(final_outputs, 1)

    def calculate_loss(self, lstm_outputs, batch_size, sequence_lengths, labels):
        logits = self.calculate_logits(lstm_outputs, batch_size, sequence_lengths)
        float_labels = tf.cast(labels, tf.float32)
        batch_loss = tf.nn.sigmoid_cross_entropy_with_logits(float_labels, logits)
        return tf.reduce_sum(batch_loss)

    def logits_to_predictions(self, logits):
        probs = tf.nn.sigmoid(logits)
        return tf.round(probs)
