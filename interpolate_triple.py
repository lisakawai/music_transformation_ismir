import os
import h5py
import pickle
from argparse import ArgumentParser
import numpy as np

import torch

import constants
from model import Classifier, FaderVAE
from arguments import generate_cfg_fader
from data_converter import RollAugMonoSingleSequenceConverter
from note_sequence_ops import steps_per_quarter_to_seconds_per_step, quantize_note_sequence
from midi_io import note_sequence_to_pretty_midi
from chords_infer import infer_chords_for_sequence
from note_sequence import Tempo, TimeSignature, InstrumentInfo

# CHANGE PATH HERE
output_path_base = 'output'


def to_tensor(data, device):
    if device == 'cpu':
        return torch.LongTensor(data, device=device)
    return torch.cuda.LongTensor(data, device=device)


def to_float_tensor(data, device):
    if device == 'cpu':
        return torch.FloatTensor(data, device=device)
    return torch.cuda.FloatTensor(data, device=device)


def test(f_chord_test, f_style_test, f_event_test, keys_test, model, device, attrs,
         batch_size):
    model.eval()
    losses, losses_d, losses_kl, accs = 0., 0., 0., 0.
    preds, org_styles, chords, interpolation_for_back = {}, {}, {}, {}

    interpolate_num = 11
    vec_to_interpolate = to_float_tensor(np.array([(i - 5) * 0.1 for i in range(interpolate_num)]), device=device)
    mul_tgt = [-0.5, 0.5]
    vec_to_interpolate_mul = to_float_tensor(np.array([[i, j, k] for i in mul_tgt for j in mul_tgt for k in mul_tgt]),
                                             device=device)

    with torch.no_grad():
        for key in keys_test:
            chord_tensor = to_float_tensor(f_chord_test[key], device=device).repeat(batch_size, 1, 1)
            event_tensor = to_tensor(f_event_test[key], device=device).repeat(batch_size, 1, 1)
            style_tensor = to_float_tensor([f_style_test[attr + '/' + key] for attr in attrs],
                                           device=device).reshape(1, len(attrs)).repeat(batch_size, 1)

            style_tensor[:interpolate_num, 0] += vec_to_interpolate
            style_tensor[1 * interpolate_num: 2 * interpolate_num, 1] += vec_to_interpolate
            style_tensor[2 * interpolate_num: 3 * interpolate_num, 2] += vec_to_interpolate
            style_tensor[3 * interpolate_num: 3 * interpolate_num + len(mul_tgt) ** 3] += vec_to_interpolate_mul

            loss, pred, lv, acc, distribution = model(event_tensor, chord_tensor, style_tensor)

            losses += loss
            accs += acc

            original_style_value_0 = float(np.array(f_style_test[attrs[0] + '/' + key]))
            original_style_value_1 = float(np.array(f_style_test[attrs[1] + '/' + key]))
            original_style_value_2 = float(np.array(f_style_test[attrs[2] + '/' + key]))
            names = ['{}_{}_{:.5}'.format(key, attrs[0], original_style_value_0 + (i - 5) * 0.1) for i in
                     range(interpolate_num)] + \
                    ['{}_{}_{:.5}'.format(key, attrs[1], original_style_value_1 + (i - 5) * 0.1) for i in
                     range(interpolate_num)] + \
                    ['{}_{}_{:.5}'.format(key, attrs[2], original_style_value_2 + (i - 5) * 0.1) for i in
                     range(interpolate_num)] + \
                    ['{}_demo_{}_{:.5}_{}_{:.5}_{}_{:.5}'.format(
                        key,
                        attrs[0], original_style_value_0 + i,
                        attrs[1], original_style_value_1 + j,
                        attrs[2], original_style_value_2 + k)
                        for i in mul_tgt for j in mul_tgt for k in mul_tgt]
            for i, name in enumerate(names):
                preds[name] = pred[i]

    return losses.item() / batch_size, accs / batch_size, preds


def evaluation(model, cfg, device):
    f_chord_test = h5py.File(cfg['data']['chord_f_valid'], 'r')
    f_event_test = h5py.File(cfg['data']['event_valid'], 'r')
    f_style_test = h5py.File(cfg['data']['attr_valid'], 'r')
    with open(cfg['data']['keys_valid'], 'rb') as f:
        keys_test = pickle.load(f)

    return test(f_chord_test, f_style_test, f_event_test, keys_test, model, device,
                cfg['attr'], cfg['batch_size'])


def run(args):
    output_dir = os.path.join(output_path_base, args.model_name)
    latest_model_text_file = os.path.join(output_dir, 'latest_model.txt')
    sample_dir = os.path.join(output_dir, 'samples')
    if not os.path.exists(sample_dir):
        os.mkdir(sample_dir)

    if not os.path.exists(output_dir) and not os.path.exists(latest_model_text_file):
        raise IOError("Model file not found.")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(torch.cuda.current_device())

    if args.model_path:
        latest_model_path = args.model_path
    else:
        with open(latest_model_text_file, 'r') as f:
            latest_model_path = f.read()
    checkpoint = torch.load(latest_model_path)
    latest_model_name = latest_model_path.split('/')[-1]
    cfg = generate_cfg_fader(None, args, output_dir, checkpoint)

    print(cfg)

    model_d = Classifier(input_dim=cfg['z_dims'],
                         num_layers=cfg['model']['discriminator']['num_layers'],
                         n_attr=len(cfg['attr']),
                         activation=cfg['activation_d'],
                         n_classes=8,
                         device=device)

    model = FaderVAE(vocab_size=cfg['vocab_size'],
                     hidden_dims=cfg['model']['encoder']['hidden_size'],
                     z_dims=cfg['z_dims'],
                     n_step=cfg['bars_per_data'] * cfg['steps_per_quarter'] * constants.QUARTERS_PER_BAR,
                     device=device,
                     n_attr=cfg['n_attr'])

    model.load_state_dict(checkpoint['model'])
    model_d.load_state_dict(checkpoint['model_d'])
    model.to(device)
    model_d.to(device)
    print(model)

    losses, accs, preds = evaluation(model, cfg, device)
    val_result_path = os.path.join(output_dir, 'eval_result.txt')
    with open(val_result_path, 'a') as f:
        f.write("model: {}, val_loss: {}, acc: {}, ".format(
            latest_model_name, losses, accs))

    RPSC = RollAugMonoSingleSequenceConverter(steps_per_quarter=cfg['steps_per_quarter'],
                                              quarters_per_bar=constants.QUARTERS_PER_BAR,
                                              chords_per_bar=cfg['chords_per_bar'],
                                              bars_per_data=cfg['bars_per_data'])

    sample_save_path = os.path.join(sample_dir, latest_model_name + '_original')
    if not os.path.exists(sample_save_path):
        os.mkdir(sample_save_path)

    seconds_per_step = steps_per_quarter_to_seconds_per_step(cfg['steps_per_quarter'], 60)

    f_chord_test = h5py.File(cfg['data']['chord_valid'], 'r')
    f_event_test = h5py.File(cfg['data']['event_valid'], 'r')
    with open(cfg['data']['keys_valid'], 'rb') as f:
        keys_test = pickle.load(f)

    chord_acc, chord_style_acc = 0., 0.
    for key, event in preds.items():
        # Normalized
        event = event.reshape(cfg['chords_per_data'], -1)
        ns = RPSC.to_note_sequence_from_events(event, seconds_per_step)  # Normalized
        tempo = Tempo(time=0., qpm=60)
        ns.tempos.append(tempo)
        ns.instrument_infos = {
            InstrumentInfo('piano', 0),
        }
        time_signature = TimeSignature(time=0, numerator=4, denominator=4)
        ns.time_signatures.append(time_signature)

        quantized_ns = quantize_note_sequence(ns, steps_per_quarter=cfg['steps_per_quarter'])
        try:
            ns_with_chord = infer_chords_for_sequence(quantized_ns, chords_per_bar=cfg['chords_per_bar'])
        except Exception as e:
            print(e)
            continue
        pm = note_sequence_to_pretty_midi(ns_with_chord)
        key_string = '_'.join(key.split('/'))
        output_path = os.path.join(sample_save_path, key_string + '.mid')

        print(output_path)
        pm.write(output_path)

        # Normalized key comparison
        chord_list = [ta_chord.text for ta_chord in ns_with_chord.text_annotations]
        chord_txt = ",".join(chord_list)
        output_chord_path = os.path.join(sample_save_path, key_string + '.txt')
        with open(output_chord_path, 'w') as f:
            f.write(chord_txt)

    with open(val_result_path, 'a') as f:
        f.write("chord_acc: {}\n".format(chord_acc / len(keys_test)))

    original_path = os.path.join(sample_dir, 'original')
    if not os.path.exists(original_path):
        os.mkdir(original_path)

        for key in keys_test:
            # Unnormalized
            ns = RPSC.to_note_sequence_from_events(np.array(f_event_test[key]), seconds_per_step)
            pm = note_sequence_to_pretty_midi(ns)
            key_string = '_'.join(key.split('/'))
            output_path = os.path.join(original_path, key_string + '.mid')
            pm.write(output_path)
            chord = list(f_chord_test[key])
            chord_list = [RPSC.chord_from_index(c) for c in chord]
            chord_list = ",".join(chord_list)
            output_chord_path = os.path.join(original_path, key_string + '.txt')
            with open(output_chord_path, 'w') as f:
                f.write(chord_list)


if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument('--gpu', type=int, default=[0, 1], nargs='+', help='used gpu')
    parser.add_argument('--model_name', type=str, default="tmp", help='model name')
    parser.add_argument('--model_path', type=str, help='to use a specific model, not latest')

    args = parser.parse_args()
    os.environ["CUDA_VISIBLE_DEVICES"] = ','.join(str(x) for x in args.gpu)

    run(args)
