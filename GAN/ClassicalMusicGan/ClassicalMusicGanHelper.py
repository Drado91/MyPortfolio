from __future__ import division
import sys
import argparse
import pretty_midi
import librosa
import numpy as np
import matplotlib.pyplot as plt
import os
import sklearn
import librosa.display
import pandas as pd
import pygame #Play midi file

os.getcwd()
midi_data = pretty_midi.PrettyMIDI(os.getcwd()+'/musicnet_midis/Bach/2186_vs6_1.mid')
os.walk(os.getcwd()+'/musicnet_midis')
[x[0] for x in os.walk(os.getcwd()+'/musicnet_midis')]
root=os.getcwd()+'/musicnet_midis'
data=[]
for path, subdirs, files in os.walk(root):
    sub_dir_str=path.split('/')[-1]
    for name in files:
        sample_path=os.path.join(path, name)
        sample_name=sub_dir_str + '_' + name.split('.')[0]
        piano_roll_matrix = pretty_midi.PrettyMIDI(sample_path).get_piano_roll(100)
        data.append([sample_name, piano_roll_matrix, sample_path])

fsr = 100 #midi sampling frequency
piano_matrix = midi_data.get_piano_roll(fs=fsr)
print(np.shape(piano_matrix))
#plt.figure(figsize = (30,30))
plt.imshow(piano_matrix[:,:1000], origin="lower")
print('finito')
def plot_piano_roll(pm, start_pitch, end_pitch, fs=100):
    # Use librosa's specshow function for displaying the piano roll
    librosa.display.specshow(pm.get_piano_roll(fs)[start_pitch:end_pitch],
                             hop_length=1, sr=fs, x_axis='time', y_axis='cqt_note',
                             fmin=pretty_midi.note_number_to_hz(start_pitch))
    plt.figure(figsize=(8, 4))
def play_midi_file(midi_path):
    def play_music(music_file):
        """
        stream music with mixer.music module in blocking manner
        this will stream the sound from disk while playing
        """
        clock = pygame.time.Clock()
        try:
            pygame.mixer.music.load(music_file)
            print("Music file %s loaded!" % music_file)
        except pygame.error:
            print("File %s not found! (%s)" % (music_file, pygame.get_error()))
            return
        pygame.mixer.music.play()
        while pygame.mixer.music.get_busy():
            # check if playback has finished
            clock.tick(30)
    midi_file = midi_path
    freq = 44100    # audio CD quality
    bitsize = -16   # unsigned 16 bit
    channels = 2    # 1 is mono, 2 is stereo
    buffer = 1024    # number of samples
    pygame.mixer.init(freq, bitsize, channels, buffer)
    # optional volume 0 to 1.0
    pygame.mixer.music.set_volume(0.8)
    try:
        play_music(midi_file)
    except KeyboardInterrupt:
        # if user hits Ctrl/C then exit
        # (works only in console mode)
        pygame.mixer.music.fadeout(1000)
        pygame.mixer.music.stop()
        raise SystemExit
