import numpy as np

from constants import STANDARD_PPQ, PITCH_CLASS_NAMES
from note_sequence import Note, NoteSequence

INS_TO_RANGE = {
        'piano': [50, 95],
    }

class RollAugMonoSingleSequenceConverter(object):
    _CHORD_KIND_INDEX = {
        '': 0,
        'm': 1,
        '7': 2,
    }
    INS_NAMES = ['piano']
    def __init__(self, steps_per_quarter, quarters_per_bar, chords_per_bar, bars_per_data, ins_to_range=INS_TO_RANGE):
        self._chords_per_data = chords_per_bar * bars_per_data
        self._steps_per_chords = steps_per_quarter * quarters_per_bar // chords_per_bar
        self._steps_per_data = steps_per_quarter * quarters_per_bar * bars_per_data
        self._notes_per_data = None
        self.reversed_chord_kind_index = {v: k for k, v in self._CHORD_KIND_INDEX.items()}
        self.INS_TO_RANGE = ins_to_range
        self.create_vocab()

    def to_tensors_instrument_separate(self, note_sequence, train=True):
        """
        Convert note sequence to tensor assuming that monophonic for each instrument.
        :param note_sequence: quantized note sequence with chord from midi
        :param train: YES if it's data for train
        :return: rhythm tensor [num_data, bars_per_data, steps_per_data],
            pitch tensor [num_data, bars_per_data, steps_per_data], chord tensor [num_data, chords_per_data]
        """
        if train:
            shifts = [i for i in range(-5, 7)]
        else:
            shifts = [0]

        # Event list shape: [12, num_chords]
        event_lists = []  # shift x num_chord
        chord_indices = []
        chord_features = []

        for i, ta_chord in enumerate(note_sequence.text_annotations):
            this_event_list, use_this_bar = self._from_quantized_sequence(
                note_sequence,
                start_step=i * self._steps_per_chords,
                end_step=(i + 1) * self._steps_per_chords,
                shift=0)
            if use_this_bar:
                this_event_indices_shifted = list(map(
                    lambda shift: self.shift_event_indices(this_event_list, shift), shifts))
                event_lists.extend(this_event_indices_shifted)

                chord_shifted = list(map(lambda shift: self.chord_index(ta_chord, shift), shifts))
                chord_indices.extend(chord_shifted)

                chord_feature_shifted = list(map(lambda shift: np.roll(ta_chord.pitch_vector, shift), shifts))
                chord_features.extend(chord_feature_shifted)
            else:
                print('no use')

        event_lists = np.array(event_lists).reshape((-1, len(shifts), self._steps_per_chords))  # num_chord, shifts, event
        event_lists = np.transpose(event_lists, (1, 0, 2))  # shifts, num_chord, len_events
        chord_indices = np.array(chord_indices).reshape((-1, len(shifts))).transpose()  # shifts, num_chord
        chord_features = np.array(chord_features).reshape((-1, len(shifts), 12))  # num_chord, shifts, pitch
        chord_features = np.transpose(chord_features, (1, 0, 2))  # shifts, num_chord, pitch

        total_data_num = event_lists.shape[1] // self._chords_per_data
        if total_data_num > 0:
            event_lists = event_lists[:, :total_data_num * self._chords_per_data]
            chord_indices = chord_indices[:, :total_data_num * self._chords_per_data]
            chord_features = chord_features[:, :total_data_num * self._chords_per_data]

        event_lists = event_lists.reshape((-1, self._chords_per_data, self._steps_per_chords))
        chord_indices = chord_indices.reshape((-1, self._chords_per_data))
        chord_features = chord_features.reshape((-1, self._chords_per_data, 12))
        return event_lists, chord_indices, chord_features, total_data_num

    def _from_quantized_sequence(self, quantized_sequence, start_step, end_step, shift):
        """Extract a list of events from the given quantized NoteSequence object.

        Within a step, new pitches are started with NOTE_ON and existing pitches are
        ended with NOTE_OFF. TIME_SHIFT shifts the current step forward in time.
        Args:
          quantized_sequence: A quantized NoteSequence instance.
          start_step: Start converting the sequence at this time step.
        Returns:
          A list of events.
        """
        # Adds the pitches which were on in the previous sequence.
        notes = [note for note in quantized_sequence.notes
                 if note.quantized_start_step < end_step and note.quantized_end_step > start_step]
        sorted_notes = sorted(notes, key=lambda note: (note.start_time, note.pitch))
        events = ['REST' for _ in range(end_step - start_step)]
        use_this_bar = False
        for note in sorted_notes:
            note_start_step = max(note.quantized_start_step - start_step, 0)
            note_end_step = min(note.quantized_end_step - start_step, end_step - start_step)
            events[note_start_step] = 'NOTEON_{}'.format(note.pitch)
            use_this_bar = True
            for i in range(note_start_step + 1, note_end_step):
                events[i] = 'CONTINUE'
        return events, use_this_bar

    def to_note_sequence_from_events(self, events, seconds_per_step):
        """
        Convert to note sequence.
        :param seconds_per_step:
        :return:
        """
        ns = NoteSequence(ticks_per_quarter=STANDARD_PPQ)
        ns.total_time = 0
        for i, event in enumerate(events):  # Each bar
            sequence_start_time = i * self._steps_per_chords * seconds_per_step
            sequence_end_time = (i + 1) * self._steps_per_chords * seconds_per_step
            ns = self._to_sequence(ns, event, seconds_per_step, sequence_start_time, sequence_end_time)
        ns.total_time = sequence_end_time
        return ns

    def _to_sequence(self, sequence, event_list, seconds_per_step, sequence_start_time, sequence_end_time):
        velocity = 60
        pitch = None
        pitch_start_step = 0

        for i, event_index in enumerate(event_list):
            event = self.vocab[int(event_index)]
            if event == 'CONTINUE':
                if pitch is not None:
                    'unexpected continue'
                continue
            if pitch is not None:
                start_time = pitch_start_step * seconds_per_step + sequence_start_time
                end_time = i * seconds_per_step + sequence_start_time
                end_time = min(end_time, sequence_end_time)
                note = Note(instrument=0, program=0, start_time=start_time, end_time=end_time,
                            pitch=pitch, velocity=velocity, is_drum=False)
                sequence.notes.append(note)
                pitch = None
            if 'NOTEON' in event:
                pitch = int(event.split('_')[1])
                pitch_start_step = i

        if pitch is not None:
            start_time = pitch_start_step * seconds_per_step + sequence_start_time
            end_time = (i + 1) * seconds_per_step + sequence_start_time
            end_time = min(end_time, sequence_end_time)
            note = Note(instrument=0, program=0, start_time=start_time, end_time=end_time,
                        pitch=pitch, velocity=velocity, is_drum=False)
            sequence.notes.append(note)
        return sequence

    def create_vocab(self):
        self.vocab = ['NOTEON_{}'.format(pitch) for pitch in range(
            self.INS_TO_RANGE[self.INS_NAMES[0]][0], self.INS_TO_RANGE[self.INS_NAMES[0]][1])]
        self.vocab.extend(['REST', 'CONTINUE'])

    def shift_event_indices(self, events, shift):
        shifted_events = list(map(lambda e: self.index_with_shift(e, shift), events))
        return shifted_events

    def index_with_shift(self, event, shift):
        if event in ['REST', 'CONTINUE']:
            return self.vocab.index(event)
        pitch = int(event.split('_')[1]) + shift
        assert self.INS_TO_RANGE[self.INS_NAMES[0]][0] <= pitch <= self.INS_TO_RANGE[self.INS_NAMES[0]][1]
        event_str = 'NOTEON_{}'.format(pitch)
        return self.vocab.index(event_str)

    def chord_index(self, ta_chord, offset):
        if isinstance(ta_chord.root, str) or ta_chord.kind not in self._CHORD_KIND_INDEX:  # No chord
            return 0
        root = (ta_chord.root + offset) % 12
        kind_ind = self._CHORD_KIND_INDEX[ta_chord.kind]
        return root * len(self._CHORD_KIND_INDEX) + kind_ind + 1

    def chord_from_index(self, chord_index):
        if chord_index == 0:
            return "N.C."
        root = (chord_index - 1) // len(self._CHORD_KIND_INDEX)
        kind_ind = (chord_index - 1) % len(self._CHORD_KIND_INDEX)
        return PITCH_CLASS_NAMES[root] + self.reversed_chord_kind_index[kind_ind]