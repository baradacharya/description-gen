"""
Single layer LSTM Architecture
Both Encoder and Decoder network has single layer of LSTM nodes varring like 256, 512, 1024, 2048

We will use generator in this code.
"""

import json
import os
from keras.models import Model
from keras.layers import Input, LSTM, Dense
import numpy as np


class Title_Des:
    input_texts = []
    target_texts = []
    val_input_texts = []
    val_target_texts = []
    max_encoder_seq_length = 300
    max_decoder_seq_length = 300
    batch_size = 128  # Batch size for training.
    epochs = 3  # 100
    latent_dim = 512  # Latent dimensionality of the encoding space.
    no_of_char = 127-32+1  #257 for <unk>

    def __init__(self):
        self.char_index = dict()
        self.index_char = dict()
        #256 total number of possible characters
        for i in range(32,127): #0 reserved for padding
            self.char_index[chr(i)] = i-32
            self.index_char[i-32] = chr(i)
        self.char_index['\t'] = 95
        self.index_char[95] = '\t'
        print (self.index_char)

    def process(self,Data):
        for data_ in Data:
            input_text = data_['title'].encode('utf-8')
            target_text = data_['text'].encode('utf-8')
            if (len(input_text) >= self.max_encoder_seq_length):
                continue

            if (len(target_text)+2 >= self.max_decoder_seq_length):
                continue
            target_text = '\t' + target_text + '\n'
            self.input_texts.append(input_text)
            self.target_texts.append(target_text)

        self.no_of_inputs = len(self.input_texts)

        self.encoder_input_data  = np.zeros((self.no_of_inputs, self.max_encoder_seq_length, self.no_of_char+1), dtype='float32')  # _,16,73
        self.decoder_input_data  = np.zeros((self.no_of_inputs, self.max_decoder_seq_length, self.no_of_char+1), dtype='float32')  # _,63,92
        self.decoder_target_data = np.zeros((self.no_of_inputs, self.max_decoder_seq_length, self.no_of_char+1),dtype='float32')  # _,63,92

        # fill the above 3 array , encoder_i/p data should be english, decoder i/p data and decoder o/p data should be same shifted by one timestep.
        for i, (input_text, target_text) in enumerate(zip(self.input_texts, self.target_texts)):

            if (i== self.no_of_inputs):
                print("some error occured.handled")
                break

            for t, char in enumerate(input_text):
                if char in self.char_index:
                    self.encoder_input_data[i, t, self.char_index[char]] = 1.

            for t, char in enumerate(target_text):
                if char in self.char_index:
                    # decoder_target_data is ahead of decoder_input_data by one timestep
                    self.decoder_input_data[i, t, self.char_index[char]] = 1.
                    if t > 0:
                        # decoder_target_data will be ahead by one timestep
                        # and will not include the start character.
                        self.decoder_target_data[i, t - 1, self.char_index[char]] = 1.

        return self.encoder_input_data,self.decoder_input_data,self.decoder_target_data

    def generateBatches(self,n_epochs, data_dir):
        for epoch in range(n_epochs):
            for fname in os.listdir(data_dir):
                fname = data_dir + '/' + fname
                print(fname)
                with open(fname) as file:
                    Data = json.load(file)
                Data_array = self.process(Data)

                index = 0
                iter = Data_array[1].shape[0] / self.batch_size

                for it in range(iter):
                    Data_x_1 = Data_array[0][index:index + self.batch_size][:]
                    Data_x_2 = Data_array[1][index:index + self.batch_size][:]
                    Data_y = Data_array[2][index:index + self.batch_size][:]
                    Data_x = [Data_x_1,Data_x_2]
                    yield Data_x,Data_y
                    index += self.batch_size

                Data_x_1 = Data_array[0][index:][:]
                Data_x_2 = Data_array[1][index:][:]
                Data_y = Data_array[2][index:][:]
                Data_x = [Data_x_1, Data_x_2]
                yield Data_x, Data_y


    def LSTM_model(self):

        # Define an input sequence and process it.
        # we could have given max_encoder_seq_length(seq_len) inplace of None while training but in inference n/w we will give single ch as input

        training_generator = self.generateBatches(self.epochs, './train')
        validation_generator = self.generateBatches(self.epochs, './val')

        #input_shape = #timestep/seq_len/M_M Units, feature_dim
        encoder_inputs = Input(shape=(None, self.no_of_char))  # _,_,73 = batch_size, (input shape)
        encoder = LSTM(self.latent_dim, return_state=True)
        encoder_outputs, state_h, state_c = encoder(encoder_inputs)  # LSTM has two states. working m/m 'h', long term m/m cell 'c'
        encoder_states = [state_h, state_c] # We discard `encoder_outputs` and only keep the states.


        # Set up the decoder, using `encoder_states` as initial state.
        decoder_inputs = Input(shape=(None, self.no_of_char))  # _,_,92.
        decoder = LSTM(self.latent_dim, return_sequences=True, return_state =True)
        decoder_outputs, _, _ = decoder(decoder_inputs, initial_state=encoder_states)
        decoder_dense = Dense(self.no_of_char, activation='softmax')
        decoder_outputs = decoder_dense(decoder_outputs)

        # Define the model that will turn `encoder_input_data` & `decoder_input_data` into `decoder_target_data`
        model = Model(inputs = [encoder_inputs, decoder_inputs], outputs= decoder_outputs)
        model.summary()
        # Run training
        model.compile(optimizer='rmsprop', loss='categorical_crossentropy')

        # Train model on dataset
        model.fit_generator(generator = training_generator,
                            steps_per_epoch= 5000, #self.no_of_inputs // batch_size,
                            epochs = self.epochs,
                            validation_data = validation_generator,
                            validation_steps= 500#len(partition['validation']) // batch_size
                            )

        del self.encoder_input_data
        del self.decoder_target_data
        del self.decoder_input_data

        # Define sampling models
        self.encoder_model = Model(inputs=encoder_inputs, outputs=encoder_states)

        decoder_state_input_h = Input(shape=(self.latent_dim,))
        decoder_state_input_c = Input(shape=(self.latent_dim,))
        decoder_states_inputs = [decoder_state_input_h, decoder_state_input_c]

        decoder_outputs, state_h, state_c = decoder(decoder_inputs, initial_state=decoder_states_inputs)
        decoder_states = [state_h, state_c]
        decoder_outputs = decoder_dense(decoder_outputs)
        self.decoder_model = Model(inputs = [decoder_inputs] + decoder_states_inputs, outputs = [decoder_outputs] + decoder_states)

    def decode_sequence(self,input_seq):
        # Encode the input as state vectors.
        states_value = self.encoder_model.predict(input_seq)

        # Generate empty target sequence of length 1.
        target_seq = np.zeros((1, 1, self.no_of_char))  # 1,1,92
        # Populate the first character of target sequence with the start character.
        target_seq[0, 0, self.char_index['\t']] = 1.

        # Sampling loop for a batch of sequences
        # (to simplify, here we assume a batch of size 1).
        stop_condition = False
        decoded_sentence = ''
        while not stop_condition:
            output_tokens, h, c = self.decoder_model.predict([target_seq] + states_value)
            #sampled_token_index = self.get_random_sample(output_tokens[0, -1, :])
            sampled_token_index = self.get_random_sample_update(output_tokens[0, -1, :])
            # Sample a token
            #sampled_token_index = np.argmax(output_tokens[0, -1, :])
            if sampled_token_index not in self.index_char:
                print sampled_token_index
                continue

            sampled_char = self.index_char[sampled_token_index]
            decoded_sentence += sampled_char


            # Exit condition: either hit max length
            # or find stop character.
            if (sampled_char == '\n' or len(decoded_sentence) > self.max_decoder_seq_length):
                stop_condition = True

            # Update the target sequence (of length 1).
            target_seq = np.zeros((1, 1, self.no_of_char))
            target_seq[0, 0, sampled_token_index] = 1.

            # Update states
            states_value = [h, c]

        return decoded_sentence

    def get_random_sample(self,lps):
        tot = -np.inf  # this is the log (total mass)
        wps = []
        for i, lp in enumerate(lps):
            wps.append([i, lp])  # temp will determine weightage, 1: normal case
            # log2(2^lp +2^tot)
            tot = np.logaddexp2(lp, tot)  # we are calculating cumilative log probability

        import random
        p = random.Random().random()
        char_ind = np.random.uniform(0.8, 1)
        #char_ind = random.Random().choice(wps)[0]
        # predict some random nuber find the coresponding interval.
        s = -np.inf  # running mass / accumulated (log) probability
        for ci, lp in wps:
            s = np.logaddexp2(s, lp)
            if p < pow(2, s - tot):
                char_ind = ci
                break
        return char_ind

    def get_random_sample_update(self,lps,k = 15):
        tot = -np.inf  # this is the log (total mass)
        wps = []
        import heapq
        freqs = []
        heapq.heapify(freqs)

        for i, lp in enumerate(lps):
            # temp will determine weightage, 1: normal case log2(2^lp +2^tot)
            #tot = np.logaddexp2(lp, tot)  # we are calculating cumilative log probability
            heapq.heappush(freqs, (lp, i))
            if len(freqs) > k:
                heapq.heappop(freqs)
        q_p = 0
        for lp,i in freqs:
            q_p += lp
            tot = np.logaddexp2(lp, tot)

        # Select top k char and then applly sampling

        import random
        #p = random.Random().random()
        p = np.random.uniform(0, q_p)
        char_ind = random.Random().choice(freqs)[1]
        # predict some random nuber find the coresponding interval.
        s = -np.inf  # running mass / accumulated (log) probability
        for lp,ci in freqs:
            s = np.logaddexp2(s, lp)
            if p < pow(2, s - tot):
                char_ind = ci
                break
        return char_ind

    def load_validation(self,data_dir = './test'):
        # k =10
        for fname in os.listdir(data_dir):
            if fname == ".DS_Store":
                continue
            # if(k== 0):
            #     break
            k = k-1
            fname = data_dir + '/' + fname
            with open(fname) as file:
                Data = json.load(file)
                for data_ in Data:
                    input_text = data_['title'].encode('utf-8')
                    target_text = data_['text'].encode('utf-8')
                    if (len(input_text) >= self.max_encoder_seq_length):
                        continue

                    if (len(target_text) + 2 >= self.max_decoder_seq_length):
                        continue
                    target_text = '\t' + target_text + '\n'
                    self.val_input_texts.append(input_text)
                    self.val_target_texts.append(target_text)

    def Inference(self):
        val_texts = self.load_validation()
        input_seq = np.zeros((1, self.max_encoder_seq_length, self.no_of_char), dtype='float32')
        file = open("res_Project.txt", 'a')

        for seq_index,input_text in enumerate(self.val_input_texts):
            # Take one sequence (part of the training test)
            # for trying out decoding.
            for t, char in enumerate(input_text):
                if char in self.char_index:
                    input_seq[0, t, self.char_index[char]] = 1.

            decoded_sentence = self.decode_sequence(input_seq)
            print('--')
            print('Input sentence:', input_text)
            print('Expected sentece:', self.val_target_texts[seq_index])
            print('Decoded sentence:', decoded_sentence)
            file.write(decoded_sentence)
            file.write("\n\n")
        file.close()



if __name__ == "__main__":

    TD = Title_Des()

    #TD.process(Data)

    TD.LSTM_model()

    TD.Inference()


#dense_1 (Dense)       (None, None, 96)      49248       lstm_2[0][0]