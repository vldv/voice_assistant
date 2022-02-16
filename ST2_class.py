# -*- coding: utf-8 -*-
"""
Created on Thu Feb 10 14:54:41 2022

@author: Victor Levy dit Vehel, victor.levy.vehel [at] gmail [dot] com
"""

# ext lib
import json
import torch
import os
import random
import numpy as np
import pandas as pd
from collections import deque
from textdistance import damerau_levenshtein as levenshtein
from transformers import Wav2Vec2ForCTC, Wav2Vec2Processor
from dotenv import load_dotenv
# int lib
from audio_capture import record
from logger import log

load_dotenv()

class assistant:
    
#%% INIT, DIV 

    def __init__(self):
        """ general initialization """
        # setup logger and internal memory
        self.log = log('log.txt')
        self.attentive = False # used for triggering word
        mem = int(os.getenv('memory_length'))
        self.audio = deque(maxlen = mem)
        self.text = deque(maxlen = mem)
        self.phonem = deque(maxlen = mem)
        self.order = deque(maxlen = mem)
        # Load sound processor
        self.log.info('loading processor...')
        self.processor = Wav2Vec2Processor.from_pretrained(os.getenv('MODEL_ID'))
        self.log.info('DONE')
        # Load model, push it to GPU if possible, and set it to EVAL mode
        self.log.info('loading model (online)...')
        try:
            self.model = Wav2Vec2ForCTC.from_pretrained(os.getenv('MODEL_ID'))
        except:
            self.log.warning('no internet connection, failback to local model...')
            self.model = Wav2Vec2ForCTC.from_pretrained(os.getenv('MODEL_path'))
        self.log.info('sending model to device...')
        self.device = "cuda:0" if torch.cuda.is_available() else "cpu"
        self.model.to(self.device)
        self.model.eval()
        self.log.info('DONE')
        # Phonem dictionnary
        self.log.info('loading phonems lookup table...')
        self.log.shush = True
        self.phon_lut = json.load(open(os.getenv('phonems_path'), 'r', encoding='utf-8'))
        self.log.shush = False
        self.log.info('DONE')
        # Orders dictionnary
        self.log.info('loading known orders...')
        ref_dataframe = pd.read_csv(os.getenv('order_path'))
        self.references_text = ref_dataframe['orders']
        self.references_command = ref_dataframe['command']
        self.references_phon = [self.text2phon(ref) for ref in self.references_text]
        self.log.info('DONE')


    def __cleanup__(self):
        """todo"""
        attrs = [attr for attr, value in self.__dict__.items()]
        for attr in attrs:
            print('deleting', attr)
            del(self.__dict__[attr])
        torch.cuda.empty_cache()
        
    
    def auth(self):
        """ authentifies user """
        user_input = None # todo
        if str(user_input) == os.getenv('password'):
            self.log.info('successful authentification.')
            return True 
        else:
            self.log.warning('Unsuccessful authentification.')
            return False
        
        
    def hextag(self, n=8):
        """ return a random hex string between 0 and 16^n. ~4e9 possibilities for n)=8 """
        return '%0{}x'.format(n) % random.randrange(16**n)
        

#%% ACTION PIPELINE

    def determine_action(self, action_code):
        """ action function """
        command = self.references_command[action_code]
        self.log.info("determining action from: {}".format(command))
        
        if command == 'activate':
            self.attentive = True
            self.signal_activation()
            self.log.info("activated, waiting for order...")
            return self.standby()
        
        elif command == 'deactivate':
            self.attentive = False
            self.log.info("deactivated, standby run closed.")
            return None
        
        elif command not in ['activate', 'deactivate'] and self.attentive:
            self.log.info("executing {}".format(command))
            self.take_action(command)
            return None
        
        else:
            self.log.warning("no action route was found, probably due to lack of prior activation.")
            return None
        
        
    def take_action(self, command):
        print("brrrrrr...")
        
        
    def signal_activation(self):
        print('biiiiip !')


#%% AUDIO PIPELINE

    def standby(self):
        """ listening pipeline, trigger actions """
        self.log.info('waiting for audio detection...')
        # wait for audio detection
        audio = record()
        self.audio.append(audio/audio.max())
        # process audio to get text
        print(self.audio[-1])
        self.text.append(self.audio2text(self.audio[-1] ))
        # lookup text to get phonems
        self.phonem.append(self.text2phon(self.text[-1] ))
        # compare phonems to database
        self.order.append(self.phon2order(self.phonem[-1] ))
        # take appropriate action
        self.determine_action(self.order[-1][-1])
    
    
    def audio2text(self, audio_array):
        """ convert float [-1, 1] audio signal of 16000 Hz to text """
        self.log.info('converting audio to text...')
        inputs = self.processor(
            audio_array, sampling_rate=16_000, return_tensors="pt", padding=True
            ).to(self.device)
        with torch.no_grad():
            logits = self.model(inputs.input_values, attention_mask=inputs.attention_mask).logits
        predicted_ids = torch.argmax(logits, dim=-1)
        text = self.processor.batch_decode(predicted_ids)[0]
        torch.cuda.empty_cache()
        self.log.info('DONE. Detected: {}'.format(text))
        return text
    
    
    def text2phon(self, text):
        """ convert text to phonems using the phonem lookup table, if possible. """
        self.log.info('converting text to phonems...')
        phonems = []
        missing = False
        for word in text.split():
            try:
                phonems += [self.phon_lut[word.lower()]]
            except:
                self.log.warning("no corresponding phonem was found for '{}'".format(word))
                missing = True
                phonems += [word.lower()]
        if missing:
            self.log.info('DONE. (no phonem missing)')
        else:
            self.log.info('DONE. (some phonem missing)')
        return "".join(phonems)
    
    
    def phon2order(self, phon):
        """ look for the closest match among known order using damerau levenshtein distance """
        self.log.info('converting phonems to order...')
        scores = np.zeros(len(self.reference_text))
        thres  = float(os.getenv('score_threshold'))
        for i, r_text, r_phon in enumerate(zip(self.references_text, self.references_phon)):
            scores[i] = levenshtein.normalized_similarity(phon, r_phon)
            self.log.info('{} similar to {} [{}] with score {}'.format(phon, r_phon, r_text, scores[i]))
        maxind = np.argmax(scores)
        maxval = scores(maxind)
        maxtext= self.references_text[maxind]
        if maxval >= thres:
            self.log.info("DONE. understood: '{}'".format(maxtext))
            return r_text[maxind], maxind
        else:
            self.log.info('no good enough match was found. (threshold: {})'.format(thres))
            return -1
    
   