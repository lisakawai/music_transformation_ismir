import pickle
import os
import h5py
import pretty_midi
import glob

from data_converter import RollAugMonoSingleSequenceConverter
from constants import QUARTERS_PER_BAR
from note_sequence_ops import quantize_note_sequence, calculate_style_feature, \
    delete_auftakt, filter_note
from chords_infer import infer_chords_for_sequence
from midi_io import midi_to_note_sequence

# CHANGE PARAM HERE
CHORDS_PER_BAR = 1
STEPS_PER_QUARTER = 4
BARS_PER_DATA = 4


def run():
    base_path = "data/original"
    h5py_dir = 'data/h5py'  # CHANGE PATH HERE
    metadata_dir = 'data/metadata'  # CHANGE PATH HERE

    if not os.path.exists(h5py_dir):
        os.mkdir(h5py_dir)
    if not os.path.exists(metadata_dir):
        os.mkdir(metadata_dir)

    data_splits = ["train", "test", "valid"]

    RPSC = RollAugMonoSingleSequenceConverter(steps_per_quarter=STEPS_PER_QUARTER, quarters_per_bar=QUARTERS_PER_BAR,
                                              chords_per_bar=CHORDS_PER_BAR, bars_per_data=BARS_PER_DATA)

    for m_i, mode in enumerate(data_splits):
        print(mode)
        path_lists = glob.glob(os.path.join(base_path, mode, '*.mid'))
        h5_path_chord = os.path.join(h5py_dir, mode + '_chord.h5')
        h5_path_event = os.path.join(h5py_dir, mode + '_event.h5')
        h5_path_sf = os.path.join(h5py_dir, mode + '_sf.h5')
        h5_path_chord_f = os.path.join(h5py_dir, mode + '_chord_f.h5')

        data_list = []
        f_chord = h5py.File(h5_path_chord, 'w')
        f_event = h5py.File(h5_path_event, 'w')
        f_sf = h5py.File(h5_path_sf, 'w')
        f_chord_f = h5py.File(h5_path_chord_f, 'w')

        for i, path in enumerate(path_lists):
            print(path)
            pm = pretty_midi.PrettyMIDI(path)
            ns = midi_to_note_sequence(pm)
            ns = delete_auftakt(ns)
            quantized_ns = quantize_note_sequence(ns, steps_per_quarter=STEPS_PER_QUARTER)
            quantized_sequence_with_chord = infer_chords_for_sequence(
                quantized_ns, chords_per_bar=CHORDS_PER_BAR, add_key_signatures=True)
            filtered_ns = filter_note(quantized_sequence_with_chord, ins_index=0)

            style_feature = calculate_style_feature(filtered_ns, num_inst=4, mono=True)
            if True:
                events, chord, chord_feature, data_num = RPSC.to_tensors_instrument_separate(
                    filtered_ns, train=mode == 'train')
                if len(events) == 0:
                    print(path, 'none event')
                    continue

            name = path.split('/')[-1]
            for j, (c, c_f, n) in enumerate(zip(chord, chord_feature, events)):
                ind_shift = j // data_num
                ind_data = j % data_num
                key = '{}_{}_{}'.format(name, ind_data, ind_shift)
                try:
                    f_chord.create_dataset(key, data=c)
                    f_sf.create_dataset(key, data=style_feature)
                    f_chord_f.create_dataset(key, data=c_f)
                    f_event.create_dataset(key, data=n)
                    data_list.append(key)
                except Exception as e:
                    print(path, e)

        data_list_path = os.path.join(metadata_dir, mode + '.pkl')
        with open(data_list_path, 'wb') as f:
            pickle.dump(data_list, f, protocol=2)

        f_chord.close()
        f_event.close()
        f_sf.close()
        f_chord_f.close()


if __name__ == "__main__":
    run()
