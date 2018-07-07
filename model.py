from keras.models import Model
from keras.layers import Input, Dense, Embedding, Lambda, TimeDistributed, Add, Conv1D, Layer
from kulc.layer_normalization import LayerNormalization
from kulc.attention import MultiHeadAttention


class PositionWiseFeedForward(Layer):
    def __init__(self, d_model=512, d_ff=2048, **kwargs):
        self._d_model = d_model
        self._d_ff = d_ff
        super().__init__(**kwargs)
        
    def build(self, input_shape):
        self._conv1 = Conv1D(self._d_ff, kernel_size=1, activation="relu")
        self._conv2 = Conv1D(self._d_model, kernel_size=1)
        super().build(input_shape)
    
    def call(self, x):
        intermediate_x = self._conv1(x)
        return self._conv2(intermediate_x)

class EncoderLayer(object):
    def __init__(self, h=8, d_k=64, d_v=64, d_model=512):
        self._mha = MultiHeadAttention(h=h, d_k=d_k, d_v=d_v, d_model=d_model)
        self._ln_a = LayerNormalization()
        self._psfw = PositionWiseFeedForward()
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
    def __init__(self, h=8, d_k=64, d_v=64, d_model=512):
        self._mha_a = MultiHeadAttention(h=h, d_k=d_k, d_v=d_v, d_model=d_model)
        self._mha_b = MultiHeadAttention(h=h, d_k=d_k, d_v=d_v, d_model=d_model)
        self._psfw = PositionWiseFeedForward()
        self._ln_a = LayerNormalization()
        self._ln_b = LayerNormalization()
        self._ln_c = LayerNormalization()
        self._add_a = Add()
        self._add_b = Add()
        self._add_c = Add()
        
    def __call__(self, x, encoder_output):
        y = self._mha_a([x, x, x])
        y = self._add_a([x, y])
        x = self._ln_a(y)
        
        y = self._mha_b([x, encoder_output, encoder_output])
        y = self._add_b([x, y])
        x = self._ln_b(y)
        
        y = self._psfw(x)
        y = self._add_c([x, y])
        x = self._ln_c(y)
        
        return x        

class Encoder(object):
    def __init__(self, embedding, n=6, h=8, d_k=64, d_v=64, d_model=512):
        self._embedding = embedding
        self._n = n
        
        self._layers = [EncoderLayer(h=h, d_k=d_k, d_v=d_v, d_model=d_model) for _ in range(n)]
    
    def __call__(self, x):
        x = self._embedding(x)
        # TODO: add positional encoding
        
        for layer in self._layers:
            x = layer(x)
            
        return x

class Decoder(object):
    def __init__(self, embedding, n=6, h=8, d_k=64, d_v=64, d_model=512):
        self._embedding = embedding
        self._n = n
        
        self._layers = [DecoderLayer(h=h, d_k=d_k, d_v=d_v, d_model=d_model) for _ in range(n)]
    
    def __call__(self, x, encoder_output):
        x = self._embedding(x)
        # TODO: add positional encoding
        
        for layer in self._layers:
            x = layer(x, encoder_output)
            
        return x

def build_transformer(vocabulary_size, n, h, d_k, d_v, d_model):
    enc_input = Input(shape=(None,))
    dec_input = Input(shape=(None,))
    word_embedding = Embedding(vocabulary_size, d_model)

    enc = Encoder(word_embedding, n, h, d_k, d_v, d_model)
    dec = Decoder(word_embedding, n, h, d_k, d_v, d_model)

    enc_output = enc(enc_input)
    dec_output = dec(dec_input, enc_output)

    lin_dense = TimeDistributed(Dense(d_model))
    fin_output = TimeDistributed(Dense(vocabulary_size, activation="softmax"))

    lin_dense_out = lin_dense(dec_output)
    fin_output_out = fin_output(lin_dense_out)

    model = Model(inputs=[enc_input, dec_input], outputs=fin_output_out)
    return model