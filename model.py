import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Normal


class FaderVAE(nn.Module):
    def __init__(self,
                 vocab_size,
                 hidden_dims,
                 z_dims,
                 n_step,
                 device,
                 n_attr,
                 k=1000):
        super(FaderVAE, self).__init__()
        self.gru_0 = nn.GRU(
            vocab_size,
            hidden_dims,
            batch_first=True,
            bidirectional=True)
        self.linear_mu = nn.Linear(hidden_dims * 2, z_dims)
        self.linear_var = nn.Linear(hidden_dims * 2, z_dims)
        self.grucell_1 = nn.GRUCell(
            z_dims + vocab_size + n_attr,
            hidden_dims)
        self.grucell_2 = nn.GRUCell(hidden_dims, hidden_dims)
        self.linear_init_1 = nn.Linear(z_dims, hidden_dims)
        self.linear_out_1 = nn.Linear(hidden_dims, vocab_size)
        self.n_step = n_step
        self.vocab_size = vocab_size
        self.hidden_dims = hidden_dims
        self.eps = 1
        self.sample = None
        self.iteration = 0
        self.z_dims = z_dims
        self.k = torch.FloatTensor([k])
        self.device = device

    def _sampling(self, x):
        idx = x.max(1)[1]
        x = torch.zeros_like(x)
        arange = torch.arange(x.size(0)).long()
        if torch.cuda.is_available():
            arange = arange.cuda()
        x[arange, idx] = 1
        return x

    def encoder(self, x, condition):
        self.gru_0.flatten_parameters()
        x = self.gru_0(x)
        x = x[-1]
        x = x.transpose_(0, 1).contiguous()
        x = x.view(x.size(0), -1)
        mu = self.linear_mu(x)
        var = self.linear_var(x).exp_()
        distribution = Normal(mu, var)
        return distribution

    def encode(self, x, condition):
        b, c, s = x.size()
        x = x.reshape(b, -1)
        x = torch.eye(self.vocab_size)[x].to(self.device)
        dis = self.encoder(x, condition)
        z = dis.rsample()
        return z, None, None

    def decoder(self, z, condition, style):
        out = torch.zeros((z.size(0), self.vocab_size))
        out[:, -1] = 1.
        x, hx = [], [None, None]
        t = torch.tanh(self.linear_init_1(z))
        hx[0] = t
        if torch.cuda.is_available():
            out = out.cuda()
        for i in range(self.n_step):
            out = torch.cat([out.float(), z, style], 1)
            hx[0] = self.grucell_1(out, hx[0])
            if i == 0:
                hx[1] = hx[0]
            hx[1] = self.grucell_2(hx[0], hx[1])
            out = F.log_softmax(self.linear_out_1(hx[1]), 1)
            x.append(out)
            if self.training:
                p = torch.rand(1).item()
                if p < self.eps:
                    out = self.sample[:, i, :]
                else:
                    out = self._sampling(out)
                self.eps = self.k / \
                    (self.k + torch.exp(self.iteration / self.k))
            else:
                out = self._sampling(out)
        return torch.stack(x, 1)

    def forward(self, x, condition, style):
        b, c, s = x.size()
        x = x.reshape(b, -1)
        x_indices = x
        x = torch.eye(self.vocab_size)[x].to(self.device)
        condition = condition.repeat(1, 1, 1, s)
        condition = condition.reshape(b, c * s, -1)
        if self.training:
            self.sample = x
            self.iteration += 1
        dis = self.encoder(x, condition)
        z = dis.rsample()
        recon = self.decoder(z, condition, style)
        preds = torch.argmax(recon, dim=-1)
        acc = torch.sum(torch.eq(preds, x_indices)).item() / (x_indices.size(0) * x_indices.size(1))
        loss = F.nll_loss(recon.reshape(-1, recon.size(-1)), x_indices.reshape(-1))
        return loss, preds, z, acc, dis


class Classifier(nn.Module):
    def __init__(self, input_dim, num_layers, n_attr, n_classes, activation, device):
        super(Classifier, self).__init__()
        if activation == 'tanh':
            activation_f = nn.Tanh
        elif activation == 'relu':
            activation_f = nn.ReLU
        elif activation == 'leakyrelu':
            activation_f = nn.LeakyReLU

        assert num_layers >= 2
        layers = []
        for _ in range(num_layers - 1):
            layers.append(nn.Linear(input_dim, input_dim // 2))
            layers.append(activation_f())
            input_dim = input_dim // 2
        layers.append(nn.Linear(input_dim, n_attr * n_classes))
        layers.append(nn.Sigmoid())
        self.layers = nn.ModuleList(layers)
        self.n_classes = n_classes
        self.n_attr = n_attr
        self.device = device
        self.ordinal_labels = torch.ones((8, 8)).triu()

    def calc_loss(self, pred, class_label, is_discriminator=True, is_ordinal=False):
        """

        :param pred: [b, num_attr]
        :param class_label: [b, num_attr]
        :param is_discriminator:
        :param is_ordinal:
        :return:
        """
        criterion = nn.BCELoss()
        if not is_ordinal:
            label = torch.eye(self.n_classes)[class_label]
        else:
            b, num_attr = class_label.size()
            class_label = class_label.reshape(-1)
            label = self.ordinal_labels[class_label]
            label = label.reshape(b, num_attr, self.n_classes)
        label = label.to(self.device)
        if not is_discriminator:
            label = - label + 1
        return criterion(pred, label)

    def forward(self, lv):
        for i in range(len(self.layers)):
            lv = self.layers[i](lv)
        lv = lv.reshape(lv.size(0), self.n_attr, self.n_classes)
        return lv
