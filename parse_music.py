from music21 import converter, instrument, stream
from music21.note import Note
from music21.chord import Chord
import numpy
import numpy as np
from keras.utils import to_categorical


class PrepareError(Exception):

    def __init__(self, message):
        super().__init__(message)


class PrepareModel:

    def __init__(self, count_notes, model_inp, model_out, notes, durations):
        self.count_notes = count_notes
        self.model_inp = model_inp
        self.model_out = model_out
        self.sorted_notes = notes
        self.sorted_durations = durations


class MidiFile:

    def __init__(self, name: str):
        self.notes = []
        self.int_notes = []
        self.name = name

    def read_file(self) -> None:
        file = converter.parse(self.name)
        parts = instrument.partitionByInstrument(file)
        if parts:
            notes_to_parse = parts.parts[0].recurse()
        else:
            notes_to_parse = file.flat.notes

        for music_note in notes_to_parse:
            if isinstance(music_note, Note):
                self.notes.append({'name': str(music_note.pitch), 'duration': str(music_note.duration.type)})
            elif isinstance(music_note, Chord):
                self.notes.append({'name': '.'.join(str(n) for n in music_note.normalOrder), 'duration': music_note.duration.type})
        print(self.notes)


class MidiFilesContainer:
    midi_files = []
    all_notes = []

    def append(self, file: MidiFile):
        self.midi_files.append(file)
        self.all_notes += file.notes

    def _notes_to_int(self) -> (dict, dict):
        sorted_notes = []
        sorted_durations = []
        for music_note in self.all_notes:
            sorted_notes.append(music_note['name'])
            sorted_durations.append(music_note['duration'])

        sorted_notes = sorted(set(sorted_notes))
        sorted_durations = sorted(set(sorted_durations))
        return (dict((music_note, number) for number, music_note in enumerate(sorted_notes)),
                dict((duration, number) for number, duration in enumerate(sorted_durations)))

    def prepare_to_model(self, seq_len) -> PrepareModel:
        model_inp = []
        model_out = []
        sorted_notes, sorted_durations = self._notes_to_int()

        _notes = [music_note['name'] for music_note in self.all_notes]
        _durations = [music_note['duration'] for music_note in self.all_notes]
        for i in range(0, len(_notes) - seq_len):
            seq_in = list(zip(_notes[i:i + seq_len], _durations[i:i + seq_len]))
            seq_out = list(zip(_notes[i + seq_len: i + 2 * seq_len], _durations[i + seq_len: i + 2 * seq_len]))
            if len(seq_in) == len(seq_out):
                model_inp.append([(sorted_notes[music_note], sorted_durations[duration]) for music_note, duration in seq_in])
                model_out.append([(sorted_notes[music_note], sorted_durations[duration]) for music_note, duration in seq_out])
            #else:
            #    raise PrepareError('len(seq_in} not eq len(seq_out)')

        model_inp = np.reshape(model_out, (len(model_inp), seq_len, 2))
        model_out = np.array(model_out)

        count_notes = max(model_inp[:, :, 0].max(), model_out[:, :, 0].max()) + 1
        network_input = to_categorical(model_inp)
        network_output = to_categorical(model_out)
        return PrepareModel(count_notes, network_input, network_output, sorted_notes, sorted_durations)


if __name__ == '__main__':
    from os import listdir
    from os.path import isfile, join
    mypath = 'music/'
    files = [f for f in listdir(mypath) if isfile(join(mypath, f))]
    container = MidiFilesContainer()

    for file in files[:3]:
        print(file)
        midi = MidiFile(mypath + file)
        midi.read_file()
        container.append(midi)

    data = container.prepare_to_model(200)
