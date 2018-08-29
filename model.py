from keras.models import Model
from keras.layers import Input, Dense, Embedding, Lambda, TimeDistributed, Add, Conv1D, Layer
from kulc.layer_normalization import LayerNormalization
from kulc.attention import MultiHeadAttention
import numpy as np
import keras.backend as K

class PositionWiseFeedForward(object):
    # def __init__(self, d_model=512, d_ff=2048, **kwargs):
    def __init__(self, d_model=512, d_ff=512, **kwargs):
        self._d_model = d_model
        self._d_ff = d_ff

        self._conv1 = Conv1D(self._d_ff, kernel_size=1, activation="relu")
        self._conv2 = Conv1D(self._d_model, kernel_size=1)
    
    def __call__(self, x):
        intermediate_x = self._conv1(x)
        return self._conv2(intermediate_x)

class EncoderLayer(object):
    def __init__(self, h=8, d_k=64, d_v=64, d_model=512, d_inner_hid=2048):
        self._mha = MultiHeadAttention(h=h, d_k=d_k, d_v=d_v, d_model=d_model)
        self._ln_a = LayerNormalization()
        self._psfw = PositionWiseFeedForward(d_model=d_model, d_ff=d_inner_hid)
        self._ln_b = LayerNormalization()
        self._add_a = Add()
        self._add_b = Add()
        
    def __call__(self, x):
        y = self._mha([x, x, x])
        y = self._add_a([x, y])
        x = self._ln_a(y)
        
        y = self._psfw(x)
        y = self._add_b([x, y])
        x = self._ln_b(y)
        
        return x         
    
class DecoderLayer(object):
	def __init__(self, h=8, d_k=64, d_v=64, d_model=512, d_inner_hid=2048, return_attention=True):
		self._mha_a = MultiHeadAttention(h=h, d_k=d_k, d_v=d_v, d_model=d_model, return_attention=return_attention)
		self._mha_b = MultiHeadAttention(h=h, d_k=d_k, d_v=d_v, d_model=d_model, return_attention=return_attention)
		self._psfw = PositionWiseFeedForward(d_model=d_model, d_ff=d_inner_hid)
		self._ln_a = LayerNormalization()
		self._ln_b = LayerNormalization()
		self._ln_c = LayerNormalization()
		self._add_a = Add()
		self._add_b = Add()
		self._add_c = Add()
		self._return_attention = return_attention
		
	def __call__(self, x, encoder_output):
		y, self_atn = self._mha_a([x, x, x])
		y = self._add_a([x, y])
		x = self._ln_a(y)
		
		y, enc_atn = self._mha_b([x, encoder_output, encoder_output])
		y = self._add_b([x, y])
		x = self._ln_b(y)
		
		y = self._psfw(x)
		y = self._add_c([x, y])
		x = self._ln_c(y)
		
		if self._return_attention:
			return [x, self_atn, enc_atn]
		else:
			return x  

class Encoder(object):
	def __init__(self, embedding, position_embedding, n=6, h=8, d_k=64, d_v=64, d_model=512, d_inner_hid=2048, null_token_value=0):
		self._embedding = embedding
		self._position_embedding = position_embedding
		self._n = n
		self._position_encoding = Lambda(_get_pos_seq, arguments={"null_token_value": null_token_value})
		
		self._layers = [EncoderLayer(h=h, d_k=d_k, d_v=d_v, d_model=d_model, d_inner_hid=d_inner_hid) for _ in range(n)]
	
	def __call__(self, x):
		x_embedded = self._embedding(x)
		pos_encoding = self._position_encoding(x)
		pos_encoding_embedded = self._position_embedding(pos_encoding)
		x = Add()([x_embedded, pos_encoding_embedded])
		
		for layer in self._layers:
			x = layer(x)
			
		return x

class Decoder(object):
	def __init__(self, embedding, position_embedding, n=6, h=8, d_k=64, d_v=64, d_model=512, d_inner_hid=2048, null_token_value=0):
		self._embedding = embedding
		self._position_embedding = position_embedding
		self._n = n
		self._position_encoding = Lambda(_get_pos_seq, arguments={"null_token_value": null_token_value})
		
		self._layers = [DecoderLayer(h=h, d_k=d_k, d_v=d_v, d_model=d_model, d_inner_hid=d_inner_hid) for _ in range(n)]
	
	def __call__(self, x, encoder_output, return_attention=False):
		x_embedded = self._embedding(x)
		pos_encoding = self._position_encoding(x)
		pos_encoding_embedded = self._position_embedding(pos_encoding)
		x = Add()([x_embedded, pos_encoding_embedded])

		self_atts = []
		enc_atts = []

		for layer in self._layers:
			x, self_att, enc_att = layer(x, encoder_output)

			if return_attention: 
				self_atts.append(self_att)
				enc_atts.append(enc_att)
		 
		if return_attention: 
			return [x, self_atts, enc_atts]
		else:
			return x

def build_transformer(source_vocabulary_size, target_vocabulary_size, max_length, share_word_embedding=False, 
                        n=6, h=8, d_k=64, d_v=64, d_model=512, optimizer="adam", null_token_value=0):
    source_input = Input(shape=(max_length,), name="source_input")
    target_input = Input(shape=(max_length,), name="target_input")

    enc_input = Lambda(lambda x:x[:,1:])(source_input)
    dec_input  = Lambda(lambda x:x[:,:-1])(target_input)
    dec_target_output = Lambda(lambda x:x[:,1:])(target_input)

    # create embedding
    source_word_embedding = Embedding(source_vocabulary_size, d_model, name="source_embedding" if share_word_embedding else "source_embedding")  # weights=[_get_positional_encoding_matrix(max_length, d_model)]
    if share_word_embedding:
        target_word_embedding = source_word_embedding
    else:
        target_word_embedding = Embedding(target_vocabulary_size, d_model, name="target_embedding")
    # embedding for the position encoding
    position_encoding = Embedding(max_length, d_model, trainable=False, weights=[_get_positional_encoding_matrix(max_length, d_model)], name="position_embedding")

    enc = Encoder(source_word_embedding, position_encoding, n=n, h=h, d_k=d_k, d_v=d_v, d_model=d_model, d_inner_hid=512)
    dec = Decoder(target_word_embedding, position_encoding, n=n, h=h, d_k=d_k, d_v=d_v, d_model=d_model, d_inner_hid=512)

    enc_output = enc(enc_input)
    dec_output = dec(dec_input, enc_output)

    # lin_dense = TimeDistributed(Dense(d_model))
    fin_output = TimeDistributed(Dense(target_vocabulary_size, activation=None, use_bias=False), name="output") # "softmax"

    # lin_dense_out = lin_dense(dec_output)
    fin_output_out = fin_output(dec_output) # lin_dense_out)

    accuracy = Lambda(_get_accuracy, arguments={"null_token_value": null_token_value})([fin_output_out, dec_target_output])
    loss = Lambda(_get_loss, arguments={"null_token_value": null_token_value})([fin_output_out, dec_target_output])

    train_model = Model(inputs=[source_input, target_input], outputs=loss)
    train_model.add_loss([loss])
    train_model.compile(optimizer, None)
    train_model.metrics_names.append('accuracy')
    train_model.metrics_tensors.append(accuracy)

    inference_model = Model([source_input, target_input], fin_output_out)

    return train_model, inference_model

def create_model(source_vocabulary_size, target_vocabulary_size, max_length, share_word_embedding=False, 
                    n=6, h=8, d_k=64, d_v=64, d_model=512, optimizer="adam", null_token_value=0):
    return build_transformer(
        source_vocabulary_size=source_vocabulary_size, target_vocabulary_size=target_vocabulary_size,
        max_length=max_length, share_word_embedding=share_word_embedding,
        n=n, h=h, d_k=d_k, d_v=d_v,d_model=d_model, optimizer=optimizer, null_token_value=null_token_value)

def _get_loss(args, null_token_value):
    y_pred, y_true = args

    y_true_id = K.cast(y_true, "int32")

    mask = K.cast(K.equal(y_true_id, null_token_value), K.floatx())
    mask = 1.0 - mask
    loss = K.sparse_categorical_crossentropy(y_true, y_pred, from_logits=True) * mask

    # take average w.r.t. the number of unmasked entries
    return K.sum(loss) / K.sum(mask)

def _get_accuracy(args, null_token_value):
    y_pred, y_true = args

    y_true = K.cast(y_true, "int32")
    mask = 1.0 - K.cast(K.equal(y_true, null_token_value), K.floatx())

    y_pred = K.cast(K.argmax(y_pred, axis=-1), "int32")
    correct = K.cast(
        K.equal(y_pred, y_true),
        K.floatx()
    )
    correct = K.sum(correct * mask, -1) / K.sum(mask, -1)

    return K.mean(correct)

def _get_pos_seq(x, null_token_value=0):
    mask = K.cast(K.not_equal(x, null_token_value), 'float32')
    pos = K.cumsum(K.ones_like(x, 'float32'), 1)
    return pos * mask

def _get_positional_encoding_matrix(max_len, d_emb):
	pos_enc = np.array([
		[pos / np.power(10000, 2 * (j // 2) / d_emb) for j in range(d_emb)] 
		if pos != 0 else np.zeros(d_emb) 
			for pos in range(max_len)
			])
	pos_enc[1:, 0::2] = np.sin(pos_enc[1:, 0::2]) # dim 2i
	pos_enc[1:, 1::2] = np.cos(pos_enc[1:, 1::2]) # dim 2i+1
	return pos_enc