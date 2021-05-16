import os
import h5py
import pickle
from argparse import ArgumentParser
import numpy as np

import torch
from torch.distributions import kl_divergence, Normal

import constants
from model import Classifier, FaderVAE
from arguments import generate_cfg_fader
from data_converter import RollAugMonoSingleSequenceConverter
from note_sequence_ops import steps_per_quarter_to_seconds_per_step, quantize_note_sequence
from midi_io import note_sequence_to_pretty_midi
from chords_infer import infer_chords_for_sequence
from note_sequence import Tempo, TimeSignature, InstrumentInfo
from utils import std_normal

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


def create_style_const(attrs, batch_size, thresholds):
    if len(attrs) == 1:
        style_ratios = [[thre] for thre in thresholds[attrs[0]]]
        names = ['_{}{:.2}'.format(attrs[0], ratio[0]) for ratio in style_ratios]
        padding = [[0.] for _ in range(batch_size - len(style_ratios) - 1)]
        style_const = np.array(style_ratios + padding)
        style_const_cls = np.array([[i] for i in range(len(thresholds[attrs[0]]))] + padding)
    elif len(attrs) == 2:
        tgt_class = [0, 4, 7]
        style_ratios = [[thresholds[attrs[0]][i], thresholds[attrs[1]][j]] for i in tgt_class for j in tgt_class]
        names = ['_{}{:.2}_{}{:.2}'.format(attrs[0], ratio[0], attrs[1], ratio[1]) for ratio in style_ratios]
        padding = [[0., 0.] for _ in range(batch_size - len(style_ratios) - 1)]
        style_const = np.array(style_ratios + padding)
        style_const_cls = np.array([[i, j] for i in tgt_class for j in tgt_class] + padding)
    elif len(attrs) == 3:
        tgt_class = [2, 5]
        style_ratios = [[thresholds[attrs[0]][i], thresholds[attrs[1]][j], thresholds[attrs[2]][k]]
                        for i in tgt_class for j in tgt_class for k in tgt_class]
        names = ['_{}{:.2}_{}{:.2}_{}{:.2}'.format(attrs[0], ratio[0], attrs[1], ratio[1], attrs[2], ratio[2])
                 for ratio in style_ratios]
        padding = [[0., 0., 0.] for _ in range(batch_size - len(style_ratios) - 1)]
        style_const = np.array(style_ratios + padding)
        style_const_cls = np.array([[i, j, k] for i in tgt_class for j in tgt_class for k in tgt_class] + padding)
    else:
        raise Exception('unsupported number of style')
    return style_const, style_const_cls, names


def get_original_style_name(attrs, original_style):
    if len(attrs) == 1:
        return '_org_{}{:.2}'.format(attrs[0], float(original_style[0][0]))
    if len(attrs) == 2:
        return '_org_{}{:.2}_{}{:.2}'.format(attrs[0], float(original_style[0][0]),
                                             attrs[1], float(original_style[0][1]))
    if len(attrs) == 3:
        return '_org_{}{:.2}_{}{:.2}_{}{:.2}'.format(attrs[0], float(original_style[0][0]),
                                                     attrs[1], float(original_style[0][1]),
                                                     attrs[2], float(original_style[0][2]))


def test(f_chord_test, f_style_test, f_style_cls_test, f_event_test, keys_test, model, model_d, device, attrs,
         batch_size, thresholds):
    model.eval()
    model_d.eval()
    losses, losses_d, losses_kl, accs = 0., 0., 0., 0.
    preds = {}

    style_const, style_const_cls, names = create_style_const(attrs, batch_size, thresholds)
    with torch.no_grad():
        for key in keys_test:
            chord_tensor = to_float_tensor(f_chord_test[key], device=device).repeat(batch_size, 1, 1)
            event_tensor = to_tensor(f_event_test[key], device=device).repeat(batch_size, 1, 1)

            original_style_value = np.array([f_style_test[attr + '/' + key] for attr in attrs]).reshape(1, -1)
            style_tensor = np.concatenate([original_style_value, style_const])
            style_tensor = to_float_tensor(style_tensor, device=device)

            original_style_cls_value = np.array([f_style_cls_test[attr + '/' + key] for attr in attrs]).reshape(1, -1)
            style_cls_tensor = np.concatenate([original_style_cls_value, style_const_cls])
            style_cls_tensor = to_tensor(style_cls_tensor, device=device)

            loss, pred, lv, acc, distribution = model(event_tensor, chord_tensor, style_tensor)
            dis_out = model_d(lv)
            loss_d = model_d.calc_loss(dis_out, style_cls_tensor)

            normal = std_normal(distribution.mean.size())
            loss_kl = kl_divergence(distribution, normal).mean()

            losses += loss
            accs += acc
            losses_d += loss_d
            losses_kl += loss_kl

            original_style_name = get_original_style_name(attrs, original_style_value)
            preds[key + original_style_name] = pred[0]
            for i, name in enumerate(names):
                preds[key + name] = pred[i + 1]

    return losses.item() / batch_size, losses_d.item() / batch_size, losses_kl.item() / batch_size, accs / batch_size,\
           preds


def evaluation(model, model_d, cfg, device):
    f_chord_test = h5py.File(cfg['data']['chord_f_valid'], 'r')
    f_event_test = h5py.File(cfg['data']['event_valid'], 'r')
    f_style_test = h5py.File(cfg['data']['attr_valid'], 'r')
    f_style_cls_test = h5py.File(cfg['data']['attr_cls_valid'], 'r')
    with open(cfg['data']['keys_valid'], 'rb') as f:
        keys_test = pickle.load(f)

    threshold_path = '/data/unagi0/kawai/Nottingham/processed_h5/roll_4_4_metadata/style_cls_thresholds.pkl'
    with open(threshold_path, 'rb') as f:
        thresholds = pickle.load(f, encoding='latin1')
    return test(f_chord_test, f_style_test, f_style_cls_test, f_event_test, keys_test, model, model_d, device,
                cfg['attr'], cfg['batch_size'], thresholds)


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

    losses, losses_d, losses_kl, accs, preds = evaluation(model, model_d, cfg, device)

    val_result_path = os.path.join(output_dir, 'eval_result.txt')
    with open(val_result_path, 'a') as f:
        f.write("model: {}, val_loss: {}, val_loss_d: {}, val_loss_kl: {}, acc: {}, ".format(
            latest_model_name, losses, losses_d, losses_kl, accs))

    RPSC = RollAugMonoSingleSequenceConverter(steps_per_quarter=cfg['steps_per_quarter'],
                                              quarters_per_bar=constants.QUARTERS_PER_BAR,
                                              chords_per_bar=cfg['chords_per_bar'],
                                              bars_per_data=cfg['bars_per_data'])

    sample_save_path = os.path.join(sample_dir, latest_model_name + '_inter')
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
