# Copyright 2019 The Magenta Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License

"""MIDI ops.
Input and output wrappers for converting between MIDI and other formats.
### THIS WORKS ONLY FOR 4/4 ####
"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import collections
import sys

import pretty_midi
import six

from note_sequence import NoteSequence, TimeSignature, KeySignature, Tempo, InstrumentInfo, Note, PitchBend, \
    ControlChange
import constants


# Allow pretty_midi to read MIDI files with absurdly high tick rates.
# Useful for reading the MAPS dataset.
# https://github.com/craffel/pretty-midi/issues/112
pretty_midi.pretty_midi.MAX_TICK = 1e10

# The offset used to change the mode of a key from major to minor when
# generating a PrettyMIDI KeySignature.
_PRETTY_MIDI_MAJOR_TO_MINOR_OFFSET = 12


class MIDIConversionError(Exception):
    pass


def midi_to_note_sequence(midi_data, fixed_instrument_infos=None):
    """Convert MIDI file contents to a NoteSequence.
    Converts a MIDI file encoded as a string into a NoteSequence. Decoding errors
    are very common when working with large sets of MIDI files, so be sure to
    handle MIDIConversionError exceptions.
    Args:
      midi_data: A string containing the contents of a MIDI file or populated
          pretty_midi.PrettyMIDI object.
    Returns:
      A NoteSequence.
    Raises:
      MIDIConversionError: An improper MIDI mode was supplied.
    """
    # In practice many MIDI files cannot be decoded with pretty_midi. Catch all
    # errors here and try to log a meaningful message. So many different
    # exceptions are raised in pretty_midi.PrettyMidi that it is cumbersome to
    # catch them all only for the purpose of error logging.

    if isinstance(midi_data, pretty_midi.PrettyMIDI):
        midi = midi_data
    else:
        try:
            midi = pretty_midi.PrettyMIDI(six.BytesIO(midi_data))
        except BaseException:
            raise MIDIConversionError('Midi decoding error %s: %s' %
                                      (sys.exc_info()[0], sys.exc_info()[1]))

    sequence = NoteSequence(ticks_per_quarter=midi.resolution)
    time_signature = TimeSignature(time=0, numerator=4, denominator=4)
    sequence.time_signatures.append(time_signature)

    # Populate key signatures.
    for midi_key in midi.key_signature_changes:
        midi_mode = midi_key.key_number // 12  # MAJOR if 0 else MINOR
        key_signature = KeySignature(
            time=midi_key.time, key=midi_key.key_number %
            12, mode=midi_mode)
        sequence.key_signatures.append(key_signature)

    # Populate tempo changes.
    tempo_times, tempo_qpms = midi.get_tempo_changes()
    for time_in_seconds, tempo_in_qpm in zip(tempo_times, tempo_qpms):
        tempo = Tempo(time=time_in_seconds, qpm=tempo_in_qpm)
        sequence.tempos.append(tempo)

    # Populate notes by gathering them all from the midi's instruments.
    # Also set the sequence.total_time as the max end time in the notes.
    midi_notes = []
    midi_pitch_bends = []
    midi_control_changes = []
    for num_instrument, midi_instrument in enumerate(midi.instruments):
        # Populate instrument name from the midi's instruments
        if fixed_instrument_infos is not None:
            num_instrument = fixed_instrument_infos.index(midi_instrument.name)
        instrument_info = InstrumentInfo(
            midi_instrument.name, num_instrument)
        sequence.instrument_infos.append(instrument_info)

        for midi_note in midi_instrument.notes:
            if not sequence.total_time or midi_note.end > sequence.total_time:
                sequence.total_time = midi_note.end
            midi_notes.append((midi_instrument.program, num_instrument,
                               midi_instrument.is_drum, midi_note))
        for midi_pitch_bend in midi_instrument.pitch_bends:
            midi_pitch_bends.append(
                (midi_instrument.program, num_instrument,
                 midi_instrument.is_drum, midi_pitch_bend))
        for midi_control_change in midi_instrument.control_changes:
            midi_control_changes.append(
                (midi_instrument.program, num_instrument,
                 midi_instrument.is_drum, midi_control_change))

    for program, instrument, is_drum, midi_note in midi_notes:
        note = Note(
            instrument,
            program,
            midi_note.start,
            midi_note.end,
            midi_note.pitch,
            midi_note.velocity,
            is_drum)
        sequence.notes.append(note)

    for program, instrument, is_drum, midi_pitch_bend in midi_pitch_bends:
        pitch_bend = PitchBend(
            instrument,
            program,
            midi_pitch_bend.time,
            midi_pitch_bend.pitch,
            is_drum)
        sequence.pitch_bends.append(pitch_bend)

    for program, instrument, is_drum, midi_control_change in midi_control_changes:
        control_change = ControlChange(
            instrument,
            program,
            midi_control_change.time,
            midi_control_change.number,
            midi_control_change.value,
            is_drum)
        sequence.control_changes.append(control_change)

    return sequence


def note_sequence_to_pretty_midi(
        sequence, drop_events_n_seconds_after_last_note=None):
    """Convert NoteSequence to a PrettyMIDI.
    Time is stored in the NoteSequence in absolute values (seconds) as opposed to
    relative values (MIDI ticks). When the NoteSequence is translated back to
    PrettyMIDI the absolute time is retained. The tempo map is also recreated.
    Args:
      sequence: A NoteSequence.
      drop_events_n_seconds_after_last_note: Events (e.g., time signature changes)
          that occur this many seconds after the last note will be dropped. If
          None, then no events will be dropped.
    Returns:
      A pretty_midi.PrettyMIDI object or None if sequence could not be decoded.
    """
    ticks_per_quarter = sequence.ticks_per_quarter or constants.STANDARD_PPQ

    max_event_time = None
    if drop_events_n_seconds_after_last_note is not None:
        max_event_time = (max([n.end_time for n in sequence.notes] or [0]) +
                          drop_events_n_seconds_after_last_note)

    # Try to find a tempo at time zero. The list is not guaranteed to be in
    # order.
    initial_seq_tempo = None
    for seq_tempo in sequence.tempos:
        if seq_tempo.time == 0:
            initial_seq_tempo = seq_tempo
            break

    kwargs = {}
    if initial_seq_tempo:
        kwargs['initial_tempo'] = initial_seq_tempo.qpm
    else:
        kwargs['initial_tempo'] = constants.DEFAULT_QUARTERS_PER_MINUTE

    pm = pretty_midi.PrettyMIDI(resolution=ticks_per_quarter, **kwargs)

    # Create an empty instrument to contain time and key signatures.
    instrument = pretty_midi.Instrument(0)
    pm.instruments.append(instrument)

    # Populate time signatures.
    for seq_ts in sequence.time_signatures:
        if max_event_time and seq_ts.time > max_event_time:
            continue
        time_signature = pretty_midi.containers.TimeSignature(
            seq_ts.numerator, seq_ts.denominator, seq_ts.time)
        pm.time_signature_changes.append(time_signature)

    # Populate key signatures.
    for seq_key in sequence.key_signatures:
        if max_event_time and seq_key.time > max_event_time:
            continue
        key_number = seq_key.key
        if constants.SCALE_MODE[seq_key.mode] == "MINOR":
            key_number += _PRETTY_MIDI_MAJOR_TO_MINOR_OFFSET
        key_signature = pretty_midi.containers.KeySignature(
            key_number, seq_key.time)
        pm.key_signature_changes.append(key_signature)

    # Populate tempos.
    for seq_tempo in sequence.tempos:
        # Skip if this tempo was added in the PrettyMIDI constructor.
        if seq_tempo == initial_seq_tempo:
            continue
        if max_event_time and seq_tempo.time > max_event_time:
            continue
        tick_scale = 60.0 / (pm.resolution * seq_tempo.qpm)
        tick = pm.time_to_tick(seq_tempo.time)
        pm._tick_scales.append((tick, tick_scale))
        pm._update_tick_to_time(0)

    # Populate instrument names by first creating an instrument map between
    # instrument index and name.
    # Then, going over this map in the instrument event for loop
    inst_infos = {}
    for inst_info in sequence.instrument_infos:
        inst_infos[inst_info.instrument] = inst_info.name

    # Populate instrument events by first gathering notes and other event types
    # in lists then write them sorted to the PrettyMidi object.
    instrument_events = collections.defaultdict(
        lambda: collections.defaultdict(list))
    for seq_note in sequence.notes:
        instrument_events[(seq_note.instrument, seq_note.program,
                           seq_note.is_drum)]['notes'].append(
                               pretty_midi.Note(
                                   seq_note.velocity, seq_note.pitch,
                                   seq_note.start_time, seq_note.end_time))
    for seq_bend in sequence.pitch_bends:
        if max_event_time and seq_bend.time > max_event_time:
            continue
        instrument_events[(seq_bend.instrument, seq_bend.program,
                           seq_bend.is_drum)]['bends'].append(
                               pretty_midi.PitchBend(seq_bend.bend, seq_bend.time))
    for seq_cc in sequence.control_changes:
        if max_event_time and seq_cc.time > max_event_time:
            continue
        instrument_events[(seq_cc.instrument, seq_cc.program,
                           seq_cc.is_drum)]['controls'].append(
                               pretty_midi.ControlChange(
                                   seq_cc.control_number,
                                   seq_cc.control_value, seq_cc.time))

    for (instr_id, prog_id, is_drum) in sorted(instrument_events.keys()):
        # For instr_id 0 append to the instrument created above.
        if instr_id > 0:
            instrument = pretty_midi.Instrument(prog_id, is_drum)
            pm.instruments.append(instrument)
        else:
            instrument.is_drum = is_drum
        # propagate instrument name to the midi file
        instrument.program = prog_id
        if instr_id in inst_infos:
            instrument.name = inst_infos[instr_id]
        instrument.notes = instrument_events[
            (instr_id, prog_id, is_drum)]['notes']
        instrument.pitch_bends = instrument_events[
            (instr_id, prog_id, is_drum)]['bends']
        instrument.control_changes = instrument_events[
            (instr_id, prog_id, is_drum)]['controls']

    return pm
