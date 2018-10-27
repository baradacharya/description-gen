from __future__ import print_function
from keras.models import Model
from keras.layers import Input, LSTM, Dense
import numpy as np

"""
hyperparameter
"""
batch_size = 64
epochs = 1
latent_dim = 256
num_samples = 10000

"""
Load Data
"""
data_path = 'hin-eng/hin.txt'
input_texts = []
input_characters = set()

output_texts = []
output_characters = set()
lines = open(data_path).read().split('\n')
for line in lines[:min(num_samples,len(lines)-1)]:
    input_text,output_text = line.split('\t')
    output_text = '\t' + output_text + '\n'
    input_texts.append(input_text)
    output_texts.append(output_text)
    for char in input_text:
        if char not in input_characters:
            input_characters.add(char)
    for char in output_text:
        if char not in output_characters:
            output_characters.add(char)

input_characters = sorted(list(input_characters))
output_characters = sorted(list(output_characters))
input_feature_dim = len(input_characters)
output_feature_dim = len(output_characters)
max_encoder_seq_length = max([len(txt) for txt in input_texts])
max_decoder_seq_length = max([len(txt) for txt in output_texts])

print('Number of samples:', len(input_texts))
print('input feature dimension:', input_feature_dim)
print('output feature dimension:', output_feature_dim)
print('Max sequence length for inputs:', max_encoder_seq_length)
print('Max sequence length for outputs:', max_decoder_seq_length)

input_char_index = dict((ch,id) for id,ch in enumerate(input_characters))
output_char_index = dict((ch, id) for id,ch in enumerate(output_characters))

encoder_input  = np.zeros((len(input_texts),max_encoder_seq_length,input_feature_dim),dtype = 'float32')
decoder_input  = np.zeros((len(output_texts),max_decoder_seq_length,output_feature_dim),dtype = 'float32')
decoder_output = np.zeros((len(output_texts),max_decoder_seq_length,output_feature_dim),dtype = 'float32')

"""
Prepare Dataset
"""
for i,(input_text,output_text) in enumerate(zip(input_texts,output_texts)):
    for time,ch in enumerate(input_text):
        encoder_input[i,time,input_char_index[ch]] = 1.
    for time,ch in enumerate(output_text):
        decoder_input[i,time, output_char_index[ch]] = 1.
        if time > 0:
            decoder_output[i, time - 1, output_char_index[ch]] = 1.

"""
Model Design
"""

"""
encoder
"""
encoder_inputs = Input(shape=(None,input_feature_dim))
encoder = LSTM(latent_dim,return_state=True)
_,state_h,state_c = encoder(encoder_inputs)
encoder_states = [state_h, state_c]

"""
decoder
"""
decoder_inputs = Input(shape=(None, output_feature_dim))
decoder = LSTM(latent_dim,return_sequences=True,return_state=True)
decoder_outputs,_,_ = decoder(decoder_inputs,initial_state = encoder_states)
decoder_dense = Dense(output_feature_dim, activation='softmax')
decoder_outputs = decoder_dense(decoder_outputs)

"""
Model
"""
model = Model([encoder_inputs, decoder_inputs], decoder_outputs)
model.summary()
model.compile(optimizer='rmsprop', loss='categorical_crossentropy')
model.fit([encoder_input, decoder_input], decoder_output,
          batch_size=batch_size,
          epochs=epochs,
          validation_split=0.2)
model.save('s2s.h5')

"""
Inference Network(Sampling)
"""
encoder_model = Model(encoder_inputs, encoder_states)

decoder_input_state_h = Input(shape=(latent_dim,))
decoder_input_state_c = Input(shape=(latent_dim,))
decoder_input_states = [decoder_input_state_h, decoder_input_state_c]
decoder_outputs, state_h, state_c = decoder(decoder_inputs, initial_state=decoder_input_states)
decoder_states = [state_h, state_c]
decoder_outputs = decoder_dense(decoder_outputs)

decoder_model = Model([decoder_inputs] + decoder_input_states,[decoder_outputs] + decoder_states)

reverse_input_char_index = dict((i, char) for char, i in input_char_index.items())
reverse_output_char_index = dict((i, char) for char, i in output_char_index.items())

def decode_sequence(input_seq):
    state_values = encoder_model.predict(input_seq)
    output_seq = np.zeros((1,1,output_feature_dim))
    output_seq[0,0,output_char_index['\t']] = 1.
    stop_condition = False
    decoded_sentence = ''
    while not stop_condition:
        output_chars,h,c = decoder_model.predict([output_seq] + state_values)

        sampled_char_index = np.argmax(output_chars[0,-1,:])
        sampled_char = reverse_output_char_index[sampled_char_index]
        decoded_sentence += sampled_char

        if (sampled_char == '\n' or len(decoded_sentence) > max_decoder_seq_length):
            stop_condition = True

        state_values = [h,c]
        output_seq[0,0,sampled_char_index] = 1.

    return decoded_sentence

for seq_index in range(10):
    input_seq = encoder_input[seq_index * 5: 5 * seq_index + 1]
    decoded_sentence = decode_sequence(input_seq)
    print('- - - - - - - - - - - - - - - - - - -')
    print('Input sentence:', input_texts[seq_index])
    print('Decoded sentence:', decoded_sentence)