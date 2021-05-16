import numpy as np
from constants import QUARTERS_PER_BAR


def generate_cfg(env_var, args, output_dir, checkpoint=None, chords_per_bar=2):
    if checkpoint:
        cfg = checkpoint['cfg']
        if 'structure' not in cfg['model']:
            cfg['model']['structure'] = 'RhythmPitchGenModel'
        if 'steps_per_quarter' not in cfg:
            cfg['steps_per_quarter'] = 4
        if 'EOR_token' not in cfg:
            cfg['EOR_token'] = cfg['steps_per_quarter'] * QUARTERS_PER_BAR // chords_per_bar + 1
            cfg['SOR_token'] = cfg['EOR_token'] + 1  # Start of rhythm
        if 'chords_per_bar' not in cfg:
            if 'EOR_token' in cfg:
                cfg['chords_per_bar'] = cfg['steps_per_quarter'] * QUARTERS_PER_BAR // (cfg['EOR_token'] - 1)
            else:
                cfg['chords_per_bar'] = chords_per_bar
        if 'use_custom_vocab' not in cfg:
            cfg['use_custom_vocab'] = False
        if 'r_alpha' not in cfg:
            cfg['r_alpha'] = 1.0
        if 'p_alpha' not in cfg:
            cfg['p_alpha'] = 1.0
        if 'offset_valid' in cfg['data']:
            cfg['data']['offset_valid'] = cfg['data']['offset_valid'].replace('vaild', 'valid')
        if 'style_valid' in cfg['data']:
            cfg['data']['style_valid'] = cfg['data']['style_valid'].replace('valid_style.h5', 'valid_sf.h5')
        if 'event_vocab_size' not in cfg['data']:
            cfg['data']['event_vocab_size'] = 1297
        if 'num_instruments' not in cfg['data']:
            cfg['data']['num_instruments'] = 5
        return cfg

    cfg = env_var
    cfg['model_name'] = args.model_name
    cfg['r_alpha'] = args.r_alpha
    cfg['p_alpha'] = args.p_alpha
    cfg['output_dir'] = output_dir
    cfg['chords_per_bar'] = chords_per_bar
    if 'steps_per_quarter' not in cfg:
        cfg['steps_per_quarter'] = 4
    if 'use_custom_vocab' not in cfg:
        cfg['use_custom_vocab'] = False
        cfg['EOR_token'] = cfg['steps_per_quarter'] * QUARTERS_PER_BAR // chords_per_bar + 1
        cfg['SOR_token'] = cfg['EOR_token'] + 1
    if args.n_iters:
        cfg['n_iters'] = args.n_iters
    if args.learning_rate:
        cfg['learning_rate'] = args.learning_rate
    if args.model_structure:
        cfg['model']['structure'] = args.model_structure
    if args.hidden_size:
        for key in cfg['model'].keys():
            if 'hidden_size' in cfg['model'][key]:
                cfg['model'][key]['hidden_size'] = args.hidden_size
    if args.num_layers:
        for key in cfg['model'].keys():
            if 'num_layers' in cfg['model'][key]:
                cfg['model'][key]['num_layers'] = args.num_layers
    if args.c_num_layers:
        cfg['model']['chord_encoder']['num_layers'] = args.c_num_layers
    if args.c_hidden_size:
        cfg['model']['chord_encoder']['hidden_size'] = args.c_hidden_size
    return cfg


def generate_cfg_fader(cfg, args, output_dir, checkpoint=None):
    if checkpoint:
        cfg = checkpoint['cfg']
        if 'steps_per_quarter' not in cfg:
            cfg['steps_per_quarter'] = 12
        if 'bars_per_data' not in cfg:
            cfg['bars_per_data'] = 4
        if 'max_shift_steps' not in cfg:
            cfg['max_shift_steps'] = 48
        if 'z_dims' not in cfg:
            cfg['z_dims'] = 128
        if 'n_classes' not in cfg:
            cfg['n_classes'] = 8
        if 'activation_d' not in cfg:
            cfg['activation_d'] = 'tanh'
        return cfg

    cfg['output_dir'] = output_dir
    cfg['z_dims'] = args.z_dims

    if args.n_iters:
        cfg['n_iters'] = args.n_iters
    if args.learning_rate:
        cfg['learning_rate'] = args.learning_rate
    if args.learning_rate_d:
        cfg['learning_rate_d'] = args.learning_rate_d
    if args.lambda_d is not None:
        cfg['lambda_d'] = args.lambda_d
    if args.lambda_kl:
        cfg['lambda_kl'] = args.lambda_kl

    if args.model_structure:
        cfg['model']['structure'] = args.model_structure
    if args.c_num_layers:
        cfg['model']['chord_encoder']['num_layers'] = args.c_num_layers
    if args.c_hidden_size:
        cfg['model']['chord_encoder']['hidden_size'] = args.c_hidden_size
    if args.e_num_layers:
        cfg['model']['encoder']['num_layers'] = args.e_num_layers
    if args.e_hidden_size:
        cfg['model']['encoder']['hidden_size'] = args.e_hidden_size
    if args.d_num_layers:
        cfg['model']['decoder']['num_layers'] = args.d_num_layers
    if args.d_hidden_size:
        cfg['model']['decoder']['hidden_size'] = args.d_hidden_size
    if args.dis_num_layers:
        cfg['model']['discriminator']['num_layers'] = args.dis_num_layers
    if args.attribute:
        cfg['attr'] = args.attribute
    if args.batch_size:
        cfg['batch_size'] = args.batch_size
    if args.is_ordinal == 1:
        cfg['is_ordinal'] = True
    else:
        cfg['is_ordinal'] = False
    cfg['thresholds'] = np.array([0.0])
    cfg['n_attr'] = len(cfg['attr'])
    return cfg