import os
import random
import time
import h5py
import pickle
from argparse import ArgumentParser
import ruamel
import ruamel.yaml

import torch
from torch import optim
from torch.utils.tensorboard import SummaryWriter
from torch.distributions import kl_divergence, Normal

from utils import time_since, std_normal
from model import Classifier, FaderVAE
from arguments import generate_cfg_fader
from constants import QUARTERS_PER_BAR

# CHANGE PATH HERE
output_path_base = 'output'
gradient_clip = 1.0


def to_tensor(data, device):
    if device == 'cpu':
        return torch.LongTensor(data, device=device)
    return torch.cuda.LongTensor(data, device=device)


def to_float_tensor(data, device):
    if device == 'cpu':
        return torch.FloatTensor(data, device=device)
    return torch.cuda.FloatTensor(data, device=device)


def train(chord_tensor, style_tensor, style_cls_tensor, event_tensor, model, model_d, optimizer, optimizer_d,
          lambda_d, lambda_kl, is_ordinal):
    """

    :param chord_tensor: [num_chord]
    :param style_tensor: [style_dim]
    :param event_tensor: [num_chord, event_max_length]
    :param model:h
    :param optimizer:
    :return: loss of rhythm, acc of rhythm, loss of pitch, acc of rhythm
    """
    model.train()
    model_d.train()

    # train discriminator
    optimizer_d.zero_grad()
    lv, _, _ = model.encode(event_tensor, chord_tensor)
    output = model_d(lv.detach())
    loss_d = model_d.calc_loss(output, style_cls_tensor, is_discriminator=True, is_ordinal=is_ordinal)
    loss_d.backward()

    torch.nn.utils.clip_grad_norm(model_d.parameters(), gradient_clip)
    optimizer_d.step()

    # train auto-encoder
    optimizer.zero_grad()
    loss_recon, _, lv, acc, distribution = model(event_tensor, chord_tensor, style_tensor)
    dis_out = model_d(lv)
    loss_d_gen = model_d.calc_loss(dis_out, style_cls_tensor, is_discriminator=False, is_ordinal=is_ordinal)

    normal = std_normal(distribution.mean.size())
    loss_kl = kl_divergence(distribution, normal).mean()
    loss = loss_recon + lambda_d * loss_d_gen + lambda_kl * loss_kl

    loss.backward()

    torch.nn.utils.clip_grad_norm(model.parameters(), gradient_clip)
    optimizer.step()

    return loss_recon.item(), loss_d.item(), loss_kl.item(), acc


def test(f_chord_test, f_style_test, f_style_cls_test, f_event_test, keys_test, model, model_d, device, attrs,
         batch_size, use_chord_vector, is_ordinal):
    model.eval()
    model_d.eval()
    losses, loss_ds, losses_kl, accs = 0., 0., 0., 0.

    with torch.no_grad():
        for i in range(0, len(keys_test), batch_size):
            key_indices = keys_test[i: i + batch_size]
            if len(key_indices) < batch_size:
                continue
            if use_chord_vector:
                chord_tensor = to_float_tensor([f_chord_test[key] for key in key_indices], device=device)
            else:
                chord_tensor = to_tensor([f_chord_test[key] for key in key_indices], device=device)
            style_tensor = to_float_tensor([[f_style_test[attr + '/' + key] for attr in attrs]
                                            for key in key_indices], device=device).reshape(batch_size, -1)
            style_cls_tensor = to_tensor([[f_style_cls_test[attr + '/' + key] for attr in attrs]
                                            for key in key_indices], device=device).reshape(batch_size, -1)
            event_tensor = to_tensor([f_event_test[key] for key in key_indices], device=device)

            loss, _, lv, acc, distribution = model(event_tensor, chord_tensor, style_tensor)
            dis_out = model_d(lv)
            loss_d = model_d.calc_loss(dis_out, style_cls_tensor, is_ordinal=is_ordinal)

            normal = std_normal(distribution.mean.size())
            loss_kl = kl_divergence(distribution, normal).mean()

            losses += loss
            loss_ds += loss_d
            losses_kl += loss_kl
            accs += acc

    batch_num = len(keys_test) // batch_size
    return losses.item() / batch_num, loss_ds.item() / batch_num, losses_kl.item() / batch_num, accs / batch_num


def train_iters(model, model_d, optimizer, optimizer_d, cfg, device, print_every=500, model_save_every=100,
                start_iter=0, use_chord_vector=False):
    start = time.time()

    log_dir = os.path.join(cfg['output_dir'], 'logs')
    writer = SummaryWriter(log_dir=log_dir)

    print_losses = []
    val_losses = []
    print_loss_total = 0  # Reset every print_every

    if use_chord_vector:
        f_chord = h5py.File(cfg['data']['chord_f'], 'r')
        f_chord_test = h5py.File(cfg['data']['chord_f_test'], 'r')
    else:
        f_chord = h5py.File(cfg['data']['chord'], 'r')
        f_chord_test = h5py.File(cfg['data']['chord_test'], 'r')

    f_style = h5py.File(cfg['data']['attr'], 'r')
    f_style_cls = h5py.File(cfg['data']['attr_cls'], 'r')
    f_event = h5py.File(cfg['data']['event'], 'r')
    with open(cfg['data']['keys'], 'rb') as f:
        keys = pickle.load(f)

    f_style_test = h5py.File(cfg['data']['attr_test'], 'r')
    f_style_cls_test = h5py.File(cfg['data']['attr_cls_test'], 'r')
    f_event_test = h5py.File(cfg['data']['event_test'], 'r')
    with open(cfg['data']['keys_test'], 'rb') as f:
        keys_test = pickle.load(f)

    for iter in range(start_iter + 1, cfg['n_iters'] + 1):
        key_indices = random.sample(keys, cfg['batch_size'])
        if use_chord_vector:
            chord_tensor = to_float_tensor([f_chord[key] for key in key_indices], device=device)
        else:
            chord_tensor = to_tensor([f_chord[key] for key in key_indices], device=device)
        style_tensor = to_float_tensor([[f_style[attr + '/' + key] for attr in cfg['attr']]
                                        for key in key_indices], device=device).reshape(cfg['batch_size'], -1)
        style_cls_tensor = to_tensor([[f_style_cls[attr + '/' + key] for attr in cfg['attr']]
                                        for key in key_indices], device=device).reshape(cfg['batch_size'], -1)
        event_tensor = to_tensor([f_event[key] for key in key_indices], device=device)

        loss, loss_d, loss_kl, acc = train(chord_tensor, style_tensor, style_cls_tensor, event_tensor, model, model_d,
                                           optimizer, optimizer_d, cfg['lambda_d'], cfg['lambda_kl'],
                                           is_ordinal=cfg['is_ordinal'])
        print("loss: {:.5}, loss_d: {:.5}, loss_kl {:.5}, acc: {:.5}".format(loss, loss_d, loss_kl, acc))
        writer.add_scalar("train/loss", loss, iter)
        writer.add_scalar("train/loss_d", loss_d, iter)
        writer.add_scalar("train/loss_kl", loss_kl, iter)
        writer.add_scalar("train/acc", acc, iter)
        print_loss_total += loss

        if iter % print_every == 0:
            print_loss_avg = print_loss_total / print_every
            print('%s (%d %d%%) %.4f' % (time_since(start, iter / cfg['n_iters']),
                                         iter, iter / cfg['n_iters'] * 100, print_loss_avg))

            print_losses.append(print_loss_avg)
            print_loss_total = 0

            # Test
            val_loss, val_loss_d, val_loss_kl, acc = test(f_chord_test, f_style_test, f_style_cls_test, f_event_test,
                                                          keys_test, model, model_d, device, cfg['attr'],
                                                          cfg['batch_size'], use_chord_vector,
                                                          is_ordinal=cfg['is_ordinal'])

            writer.add_scalar("test/loss", val_loss, iter)
            writer.add_scalar("test/loss_d", val_loss_d, iter)
            writer.add_scalar("test/loss_kl", val_loss_kl, iter)
            writer.add_scalar("test/acc", acc, iter)
            val_losses.append(val_loss)
            print("val_loss: {:.5}, loss_d: {:.5}, loss_kl: {:.5}, acc: {:.5}".format(
                val_loss, val_loss_d, val_loss_kl, acc))

        if iter % model_save_every == 0:
            save_path = os.path.join(cfg['output_dir'], 'models', 'checkpoint_{}'.format(iter))
            torch.save({
                'model': model.state_dict(),
                'model_d': model_d.state_dict(),
                'optimizer': optimizer.state_dict(),
                'optimizer_d': optimizer_d.state_dict(),
                'cfg': cfg,
            }, save_path)
            latest_model_text_file = os.path.join(cfg['output_dir'], 'latest_model.txt')
            with open(latest_model_text_file, 'w') as f:
                f.write(save_path)

    f_chord.close()
    f_style.close()
    f_event.close()
    f_chord_test.close()
    f_style_test.close()
    f_event_test.close()
    writer.close()


def run(args):
    yaml_path = 'env.yml'
    yaml = ruamel.yaml.YAML()
    with open(yaml_path) as stream:
        env_var = yaml.load(stream)

    output_dir = os.path.join(output_path_base, args.model_name)
    model_dir = os.path.join(output_dir, 'models')
    latest_model_text_file = os.path.join(output_dir, 'latest_model.txt')

    checkpoint = None
    start_iter = 0
    if not os.path.exists(output_dir):
        os.mkdir(output_dir)
        os.mkdir(model_dir)
    else:
        if os.path.exists(latest_model_text_file):
            with open(latest_model_text_file, 'r') as f:
                latest_model_path = f.read()
            checkpoint = torch.load(latest_model_path)
            start_iter = int(latest_model_path.split('_')[-1])

    cfg = generate_cfg_fader(env_var, args, output_dir, checkpoint)
    print(cfg)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model_d = Classifier(input_dim=cfg['z_dims'],
                         num_layers=cfg['model']['discriminator']['num_layers'],
                         n_attr=len(cfg['attr']),
                         activation=cfg['activation_d'],
                         n_classes=cfg['n_classes'],
                         device=device)

    use_chord_vector = True
    model = FaderVAE(vocab_size=cfg['vocab_size'],
                            hidden_dims=cfg['model']['encoder']['hidden_size'],
                            z_dims=cfg['z_dims'],
                            n_step=cfg['bars_per_data'] * cfg['steps_per_quarter'] * QUARTERS_PER_BAR,
                            device=device,
                            n_attr=cfg['n_attr'])

    optimizer = optim.Adam(model.parameters(), lr=cfg['learning_rate'])
    optimizer_d = optim.Adam(model_d.parameters(), lr=cfg['learning_rate_d'])

    if checkpoint:
        model.load_state_dict(checkpoint['model'])
        model_d.load_state_dict(checkpoint['model_d'])
        optimizer.load_state_dict(checkpoint['optimizer'])
        optimizer_d.load_state_dict(checkpoint['optimizer_d'])
        if device != 'cpu':
            for state in optimizer.state.values():
                for k, v in state.items():
                    if torch.is_tensor(v):
                        state[k] = v.cuda()
            for state in optimizer_d.state.values():
                for k, v in state.items():
                    if torch.is_tensor(v):
                        state[k] = v.cuda()
    model.to(device)
    model_d.to(device)

    print(model)
    train_iters(model, model_d, optimizer, optimizer_d, cfg, device, print_every=1000, model_save_every=1000,
                start_iter=start_iter, use_chord_vector=use_chord_vector)


if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument('--gpu', type=int, default=[0, 1], nargs='+', help='used gpu')
    parser.add_argument('--model_name', type=str, default="tmp", help='model name')
    parser.add_argument('--n_iters', type=int, help='epoch number')
    parser.add_argument('--learning_rate', type=float, help='learning rate')
    parser.add_argument('--learning_rate_d', type=float, help='learning rate for discriminator')
    parser.add_argument('--model_structure', type=str, help='model structure')
    parser.add_argument('--env', type=str, default='default', help='env yaml file')
    parser.add_argument('--c_num_layers', type=int, help='layer number for chord encoder')
    parser.add_argument('--c_hidden_size', type=int, help='hidden size for chord encoder')
    parser.add_argument('--e_num_layers', type=int, help='number of layers for encoder')
    parser.add_argument('--e_hidden_size', type=int, help='hidden size for encoder')
    parser.add_argument('--d_num_layers', type=int, help='number of layers for decoder')
    parser.add_argument('--d_hidden_size', type=int, help='hidden size for decoder')
    parser.add_argument('--dis_num_layers', type=int, help='number of layers for discriminator')
    parser.add_argument('--lambda_d', type=float, help='Loss weight for discriminator')
    parser.add_argument('--lambda_kl', type=float, help='Loss weight for kl')
    parser.add_argument('--attribute', type=str, nargs='*', help='attributes for style')
    parser.add_argument('--z_dims', type=int, default=128, help='dimension of latent vector')
    parser.add_argument('--batch_size', type=int, help='batch size')
    parser.add_argument('--is_ordinal', type=int, help='to use ordinal classification')

    args = parser.parse_args()
    os.environ["CUDA_VISIBLE_DEVICES"] = ','.join(str(x) for x in args.gpu)

    run(args)
