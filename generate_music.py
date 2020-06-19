from music21.note import Note
from music21.chord import Chord
from music21 import instrument, stream
import numpy as np
from os import listdir
from os.path import isfile, join
from parse_music import MidiFilesContainer, MidiFile
from keras.models import load_model
from music21.duration import Duration


class Music:

    def __init__(self, sorted_notes, sorted_durations, model_inp, model):
        self.sorted_notes = sorted_notes
        self.sorted_durations = sorted_durations
        self.model_inp = model_inp
        self.model = model
        self.start = np.random.randint(0, len(self.model_inp)-1)

    def generate(self, seq_len, a_par=0):
        pattern = self.model_inp[self.start]
        prediction_output = []
        for note_index in range(seq_len):
            prediction_input = pattern.reshape(1, seq_len, 2, len(self.sorted_notes))
            prediction_input = prediction_input / float(len(self.sorted_notes))
            predictions = self.model.predict(prediction_input, verbose=0)[0]
            for prediction in predictions:
                index = np.argmax(prediction[0])
                duration_i = np.argmax(prediction[1])

                for name, value in self.sorted_notes.items():
                    if value == index:
                        result = name
                        break
                    else:
                        result = None

                for name, value in self.sorted_durations.items():
                    if value == duration_i:
                        duration = name
                        break
                    else:
                        duration = None

                prediction_output.append((result, Duration(duration)))
                result = np.zeros_like(prediction)
                result[0][index] = 1
                result[1][duration_i] = 1
                pattern = np.concatenate([pattern, [result]])
            pattern = pattern[len(pattern) - seq_len:len(pattern)]

        offset = 0
        output_notes = []
        for pattern, duration in prediction_output:
            if pattern.isdigit() or ('.' in pattern):
                notes_in_chord = pattern.split('.')
                notes = []
                for current_note in notes_in_chord:
                    new_note = Note(int(current_note))
                    new_note.duration = duration
                    new_note.storedInstrument = instrument.PanFlute()
                    notes.append(new_note)
                new_chord = Chord(notes)
                new_chord.offset = offset
                output_notes.append(new_chord)
            else:
                new_note = Note(pattern)
                new_note.offset = offset
                new_note.storedInstrument = instrument.Flute()
                output_notes.append(new_note)
            offset += 0.6

        midi_stream = stream.Stream(output_notes)
        midi_stream.write('midi', fp=f'my_music/{self.model.name}_{self.start}.mid')


seq_len = 50

EPOCH = 2
MODEL_FILE_NAME = f'models/Model_Epoch{EPOCH}.h5'
mypath = 'music/'
files = [f for f in listdir(mypath) if isfile(join(mypath, f))]

if __name__ == '__main__':
    container = MidiFilesContainer()
    for file in files:
        midi = MidiFile(mypath + file)
        midi.read_file()
        container.append(midi)
    data = container.prepare_to_model(seq_len)
    model = load_model(MODEL_FILE_NAME)
    new_music = Music(data.sorted_notes, data.sorted_durations, data.model_inp, model)
    new_music.generate(seq_len, -5)
