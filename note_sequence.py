"""
Classes of note sequence.
"""
import constants


class NoteSequence(object):
    def __init__(self, ticks_per_quarter):
        self.ticks_per_quarter = ticks_per_quarter
        self.time_signatures = []
        self.key_signatures = []
        self.tempos = []
        self.instrument_infos = []
        self.total_time = None
        self.notes = []
        self.pitch_bends = []
        self.control_changes = []
        self.text_annotations = []
        self.quantization_info = QuantizationInfo()
        self.total_quantized_steps = None


class TimeSignature(object):
    def __init__(self, time, numerator, denominator):
        """
        Ex. 6/8, numerator = 6, denominator = 8.
        :param time: Starting time of this object.
        :param numerator: Beats per measure.
        :param denominator: The type of beat.
        """
        self.time = time
        self.numerator = numerator
        self.denominator = denominator


class KeySignature(object):
    def __init__(self, time, key, mode):
        self.time = time
        self.key = key
        self.mode = mode

    def __repr__(self):
        return "time: {}, key: {}, mode: {}".format(self.time, self.key, self.mode)


class Tempo(object):
    def __init__(self, time, qpm):
        self.time = time
        self.qpm = qpm


class InstrumentInfo(object):
    def __init__(self, name, instrument):
        self.name = name
        self.instrument = instrument

    def __repr__(self):
        return "name: {}, instrument: {}".format(self.name, self.instrument)


class TextAnnotation(object):
    def __init__(self, time, quantized_step, text, annotation_type, root, kind, pitch_vector):
        self.time = time
        self.quantized_step = quantized_step
        self.text = text
        self.annotation_type = annotation_type
        self.root = root
        self.kind = kind
        self.pitch_vector = pitch_vector

    def __repr__(self):
        return "time: {}, step: {}, text: {}, type: {}, {}, {}, pitch: ".format(
            self.time, self.quantized_step, self.text, self.annotation_type, self.root, self.kind, self.pitch_vector)


class AnnotationType(object):
    CHORD_SYMBOL = 1


class Note(object):
    def __init__(self, instrument, program, start_time, end_time, pitch, velocity, is_drum):
        self.instrument = instrument
        self.program = program
        self.start_time = start_time
        self.end_time = end_time
        self.pitch = pitch
        self.velocity = velocity
        self.is_drum = is_drum
        self.quantized_start_step = None
        self.quantized_end_step = None

    def __repr__(self):
        return "instrument: {}, start: {}, end: {}, pitch: {}, qstart: {}, qend: {}\n".format(
            self.instrument, self.start_time, self.end_time, self.pitch,  self.quantized_start_step,
            self.quantized_end_step)


class PitchBend(object):
    def __init__(self, instrument, program, time, pitch, is_drum):
        self.instrument = instrument
        self.program = program
        self.time = time
        self.pitch = pitch
        self.is_drum = is_drum


class ControlChange(object):
    def __init__(self, instrument, program, time, control_number, control_value, is_drum):
        self.instrument = instrument
        self.program = program
        self.time = time
        self.control_number = control_number
        self.control_value = control_value
        self.is_drum = is_drum


class QuantizationInfo(object):
    def __init__(self):
        self.steps_per_quarter = None


class PerformanceEvent(object):
    # Start of a new note.
    NOTE_ON = 1
    # End of a note.
    NOTE_OFF = 2
    # Shift time forward.
    TIME_SHIFT = 3

    def __init__(self, event_type, event_value):
        if event_type in (PerformanceEvent.NOTE_ON, PerformanceEvent.NOTE_OFF):
            if not constants.MIN_MIDI_PITCH <= event_value <= constants.MAX_MIDI_PITCH:
                raise ValueError('Invalid pitch value: %s' % event_value)
        elif event_type == PerformanceEvent.TIME_SHIFT:
            if not 0 <= event_value:
                raise ValueError('Invalid time shift value: %s' % event_value)
        else:
            raise ValueError('Invalid event type: %s' % event_type)

        self.event_type = event_type
        self.event_value = event_value

    def __repr__(self):
        return 'PerformanceEvent(%r, %r)\n' % (self.event_type, self.event_value)

    def __eq__(self, other):
        if not isinstance(other, PerformanceEvent):
            return False
        return self.event_type == other.event_type and self.event_value == other.event_value

    def __hash__(self):
        return int(self.event_type * 300 + self.event_value)


class Instrument(object):
    Piano = 0
    Guitar = 1
    Bass = 2
    Strings = 3
    Drums = 4


class PerformanceEventWithInstrument(object):
    # Start of a new note.
    NOTE_ON = 1
    # End of a note.
    NOTE_OFF = 2
    # Shift time forward.
    TIME_SHIFT = 3
    # Both ends of sequence.
    ENDS_OF_SEQ = 4
    # Padding.
    PADDING = 5

    def __init__(self, event_type, event_value, instrument=None):
        if event_type in (PerformanceEventWithInstrument.NOTE_ON, PerformanceEventWithInstrument.NOTE_OFF):
            if not constants.MIN_MIDI_PITCH <= event_value <= constants.MAX_MIDI_PITCH:
                raise ValueError('Invalid pitch value: %s' % event_value)
            if instrument is None:
                raise ValueError('Instrument should not be empty.')
        elif event_type == PerformanceEventWithInstrument.TIME_SHIFT:
            if not 0 <= event_value:
                raise ValueError('Invalid time shift value: %s' % event_value)
        elif event_type in [PerformanceEventWithInstrument.ENDS_OF_SEQ, PerformanceEventWithInstrument.PADDING]:
            pass
        else:
            raise ValueError('Invalid event type: %s' % event_type)

        self.event_type = event_type
        self.event_value = event_value
        self.instrument = instrument

    def __repr__(self):
        return 'PerformanceEvent(%r, %r)\n' % (self.event_type, self.event_value)

    def __eq__(self, other):
        if not isinstance(other, PerformanceEventWithInstrument):
            return False
        return self.event_type == other.event_type and self.event_value == other.event_value and \
               self.instrument == self.instrument

    def __hash__(self):
        if self.instrument:
            return int(self.event_type * 1000 + self.event_value + (self.instrument + 1) * 10000)
        else:
            return int(self.event_type * 1000 + self.event_value)

