import tensorflow as tf
import tensorflow.nn.rnn_cell as rnn
tf_fc = tf.contrib.feature_column
tf_s2s = tf.contrib.seq2seq

class Seq2SeqModel(object):
    def __init__(self, vocab_size, num_lstm_layers, num_lstm_units):
        self.vocab_size = vocab_size
        self.extended_vocab_size = vocab_size + 2
        self.num_lstm_layers = num_lstm_layers
        self.num_lstm_units = num_lstm_units
        sefl.tokenizer = tf.keras.preprocessing.text.Tokenizer(num_words=vocab_size)

    def training_tuple(self, input_sequence, output_sequence):
        truncate_front = output_sequence[1:]
        truncate_back = output_sequence[:-1]
        sos_token = [self.vocab_size]
        eos_token = [self.voab_size + 1]
        input_sequence = sos_token + input_sequence + eos_token
        ground_truth = truncate_back + sos_token
        final_sequence = truncate_front + eos_token
        return input_sequence, ground_truth, final_sequence

    def lstm_cell(self, dropout_keep_prob, num_units):
        cell = rnn.LSTMCell(num_units)
        return rnn.DropoutWrapper(cell, output_keep_prob=dropout_keep_prob)

    def stacked_lstm_cell(self, is_training, num_units):
        dropout_keep_prob = 0.5 if is_training else 1.0
        cell = rnn.MultiRNNCell([self.lstm_cell(dropout_keep_probs, num_units)
                                for i in range(self.num_lstm_layers)])
        return cell

    def get_embeddings(self, sequences, scope_name):
        with tf.variable_scope(scope_name):
            cat_column = tf_fc.sequence_categorical_column_with_identity('sequences', 
                    self.extended_vocab_size)
            embedding_column = tf.feature_column.embedding_column(cat_column, 
                    int(self.extended_vocab_size ** 0.25))
            seq_dict = {'sequences': sequences}
            embeddings, sequence_lengths = tf_fc.sequence_input_layer(seq_dict, 
                    [embedding_column])
            return embeddings, tf.cast(sequence_lengths, tf.int32)

    def encoder(self, encoder_inputs, is_training):
        input_embeddings, input_seq_lens = self.get_embeddings(encoder_inputs, 'encoder_embed')
        cell_fw = self.stacked_lstm_cell(is_training, self.num_lstm_units)
        cell_bw = self.stacked_lstm_cell(is_training, self.num_lstm_units)
        enc_outputs, final_states = tf.nn.bidirectional_dynamic_rnn(
                cell_fw, cell_bw, input_embeddings, input_seq_lens, tf.float32)
        states_fw, states_bw = final_states
        
        combined_state = []
        for i in range(self.num_lstm_layers):
            bi_state_c = tf.concat([states_fw[i].c, states_bw[i].c], axis=-1)
            bi_state_h = tf.concat([states_fw[i].h, states_bw[i].h], axis=-1)
            bi_lstm_state = rnn.LSTMStateTuple(bi_state_c, bi_state_h)
            combined_state.append(bi_lstm_state)

        final_state = tuple(combined_state)
        return enc_outputs, input_seq_lens, final_state

    def decoder_cell(self, enc_outputs, input_seq_lens, is_training):
        num_decode_units = self.num_lstm_units * 2
        dec_cell = self.stacked_lstm_cell(is_training, num_decode_units)
        combined_enc_outputs = tf.conat([enc_outputs[0], enc_outputs[1]], -1)
        attention_mechanism = tf_s2s.LuongAttention(num_decode_units, combined_enc_outputs, 
                memory_sequence_length=input_seq_lens)
        return tf_s2s.AttentionWrapper(dec_cell, attention_mechanism, 
                attention_layer_size=num_decode_units)

    def decoder_helper(self, decoder_inputs, is_training, batch_size):
        if is_training:
            dec_embeddings, dec_seq_lens = self.get_embeddings(decoder_inputs, 'decoder_embed')
            helper = tf_s2s.TrainingHelper(dec_embeddings, dec_seq_lens)
        else:
            DEC_EMB_SCOPE = 'decoder_embed/sequence_input_layer/sequences_embedding'
            with tf.variable_scope(DEC_EMB_SCOPE):
                embedding_weights = tf.get_variable(
                        'embedding_weights', 
                        shape=(self.extended_vocab_size, int(self.extended_vocab_size**0.25)))
            start_tokens = tf.tile([self.vocab_size], [batch_size])
            end_token = self.vocab_size + 1
            helper = tf_s2s.GreedyEmbeddingHelper(embedding_weights, start_tokens, end_token)
            dec_seq_lens = None
        return helper, dec_seq_lens

    def decoder(self, enc_outputs, input_seq_lens, final_state, batch_size,
                decoder_inputs=None, maximum_iterations=None):
        is_training = decoder_inputs is not None
        dec_cell = self.decoder_cell(enc_outputs, input_seq_lens, is_training)
        helper, dec_seq_lens = self.decoder_helper(decoder_inputs, is_training, batch_size)
        projection_layer = tf.layers.Dense(extended_vocab_size)
        zero_cell = dec_cell.zero_state(batch_size, tf.float32)
        initial_state = zero_cell.clone(cell_state=batch_size)
        decoder = tf_s2s.BasicDecoder(dec_cell, helper, initial_state, 
                output_layer=projection_layer)
        dec_outputs = tf_s2s.dynamic_decode(decoder, maximum_iterations)[0]
        if is_training:
            return (dec_ouputs.rnn_output, dec_seq_lens)
        return dec_outputs.sample_id

    def calculate_loss(self, logits, dec_seq_lens, decoder_outputs, batch_size):
        binary_sequences = tf.sequence_mask(dec_seq_lens, tf.float32)
        batch_loss = tf.nn.sparse_softmax_cross_entropy_with_logits(decoder_outputs, logits)
        unpadded_loss = batch_loss * binary_sequences
        return tf.reduce_sum(unpadded_loss) / batch_size
