import torch
from torch.utils import data
import torchvision
import sys
from copy import deepcopy
from visdom import Visdom

from dataset import DeepmoonDataset
from dcgan_inst import Generator, Discriminator

# CUDA for PyTorch
use_cuda = torch.cuda.is_available()
device = torch.device("cuda:0" if use_cuda else "cpu")
Tensor = torch.cuda.FloatTensor if use_cuda else torch.FloatTensor

print("Using GPU:" + str(use_cuda))

# Create Visdom line plot
viz = Visdom(port=8098)
viz.close()

optsD = dict(
    title='DCGAN Discriminator Training Loss',
    xlabel='Epoch',
    width=1500,
    height=500,
)

winD = viz.line(
    Y=Tensor([0]),
    X=Tensor([0]),
    opts=optsD
)

optsG = dict(
    title='DCGAN Generator Training Loss',
    xlabel='Epoch',
    width=1500,
    height=500,
)

winG = viz.line(
    Y=Tensor([0]),
    X=Tensor([0]),
    opts=optsG
)

optsValid = dict(
    title='DCGAN Valid Solutions',
    xlabel='Epoch',
    width=1500,
    height=500,
)

winValid = viz.line(
    Y=Tensor([0]),
    X=Tensor([0]),
    opts=optsValid
)

optsValidHist = dict(
    title='DCGAN Valid Solutions Historical Average',
    xlabel='Epoch',
    width=1500,
    height=500,
)

winValidHist = viz.line(
    Y=Tensor([0]),
    X=Tensor([0]),
    opts=optsValidHist
)

# Fix random seed
seed = int(sys.argv[2]) * 128

torch.backends.cudnn.benchmark = False
torch.backends.cudnn.deterministic = True
torch.manual_seed(seed)
if use_cuda:
    torch.cuda.manual_seed_all(seed)

# Parameters
batch_size = 128
latent_size = 64
n_features = 32
lr = 0.0002
betas = (0.5, 0.999)
n_epochs = 1000
n_critic = 5
n_print = 137
gamma = 0.8

# Parameters for the constrains
start_max = 2
end_max = 1
start_min = 2
end_min = 1
sigma = 0
lmbda = 10

# Load data and split into training and validation sets
json_path = sys.argv[1]
validation_split = 0.2

dataset = DeepmoonDataset(json_path)

split_len = [round((1-validation_split) * len(dataset)), round(validation_split * len(dataset))]
split = {x: y for x, y in zip(['train', 'val'], data.random_split(dataset, lengths=split_len))}

loader = {x: data.DataLoader(split[x], batch_size, shuffle=True, num_workers=4) for x in ['train', 'val']}

# Create the GAN
netG = Generator(latent_size, n_features)
netD = Discriminator(n_features)
netG.to(device)
netD.to(device)

optimizerD = torch.optim.Adam(netD.parameters(), lr, betas)
optimizerG = torch.optim.Adam(netG.parameters(), lr, betas)

# Define penalties
class RoundNoGradient(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x):
        return x.round()
    @staticmethod
    def backward(ctx, g):
        return g

def channel_penalty(x_fake, start_max, end_max, start_min, end_min):
    count = torch.sum(RoundNoGradient.apply(x_fake), (2, 3))

    count_start = count[:, 0]
    count_end = count[:, 2]

    g_start_max = torch.relu(count_start - start_max) ** 2
    g_end_max = torch.relu(count_end - end_max) ** 2

    g_start_min = torch.relu(-count_start + start_min) ** 2
    g_end_min = torch.relu(-count_end + end_min) ** 2

    return (g_start_max + g_end_max + g_start_min + g_end_min).mean()

def duplicate_penalty(x_fake):
    count = torch.sum(RoundNoGradient.apply(x_fake), 1, keepdim=True)
    g = torch.sum(torch.relu(count - 1) ** 2, (2, 3))
    return g.mean()

def gradient_penalty(D, x_real, x_fake):
    batch_size = x_real.size(0)

    epsilon = torch.rand(batch_size, 1, 1, 1, device=device)
    x_merged = (epsilon * x_real + (1 - epsilon) * x_fake).requires_grad_(True)
    out_merged = D(x_merged)

    grad, = torch.autograd.grad(
        outputs=out_merged,
        inputs=x_merged,
        grad_outputs=Tensor(batch_size, 1).fill_(1),
        create_graph=True,
        retain_graph=True,
        only_inputs=True
    )

    return ((grad.view(batch_size, -1).norm(2, dim=1) - 1)**2).mean()

def avg_start_end(x_fake, start_max, end_max, start_min, end_min):
    count = torch.sum(x_fake.round(), (2, 3))

    count_start = count[:, 0]
    count_end = count[:, 2]

    count_start.apply_(lambda x: 1 if start_max >= x >= start_min else 0)
    count_end.apply_(lambda x: 1 if end_max >= x >= end_min else 0)

    valid_solutions = Tensor([1 if s == 1 and e == 1 else 0
                              for s, e in zip(count_start, count_end)])

    return count_start.mean(), count_end.mean(), valid_solutions.mean()

# Start training
fixed_noise = torch.randn(128, latent_size, 1, 1, device=device)

history = {'lossD': [], 'lossG': [], 'valLossD': [], 'avgStart': [], 'avgEnd': [],
           'valid': [], 'validHist': [], 'imgs': [],'epoch': []}

for epoch in range(n_epochs):
    for i, data in enumerate(loader['train']):

        x_real = data['moves'].type(Tensor)

        if epoch == 0 and i == 0:
            viz.images(
                x_real[:16].flip(2),
                opts=dict(title='Epoch' + str(epoch), width=1000, height=250)
            )

        netD.zero_grad()

        z = torch.randn(x_real.size(0), latent_size, 1, 1, device=device)
        x_fake = netG(z)

        out_real = netD(x_real)
        out_fake = netD(x_fake)

        lossD = torch.mean(out_fake) - torch.mean(out_real) \
                + lmbda * gradient_penalty(netD, x_real.data, x_fake.data) \

        lossD.backward(retain_graph=True)
        optimizerD.step()

        if i % n_critic == 0:

            netG.zero_grad()

            x_fake = netG(z)
            out_fake = netD(x_fake)

            lossG = -torch.mean(out_fake) \
                    #+ sigma * channel_penalty(x_fake, start_max, end_max, start_min, end_min) \
                    #+ sigma * duplicate_penalty(x_fake) \

            lossG.backward()
            optimizerG.step()

        # Print training stats
        if i % n_print == 0:
            print(
                "[Epoch {:5}/{:5}] [Batch {:3}/{:3}] [D loss: {:2.6f}] [G loss: {:2.6f}]".format(
                    epoch, n_epochs, i, len(loader['train']), lossD.item(), lossG.item()
                )
            )

    # Validation
    valLossD, val_size = 0, 0
    for data in loader['val']:
        x_real = data['moves'].type(Tensor)

        z = torch.randn(x_real.size(0), latent_size, 1, 1, device=device)
        x_fake = netG(z)

        out_real = netD(x_real)
        out_fake = netD(x_fake)

        valLossD += torch.mean(out_fake) - torch.mean(out_real) \
                  + lmbda * gradient_penalty(netD, x_real.data, x_fake.data)

        val_size += 1

    valLossD /= val_size

    #sigma = 0 if (epoch // 20) % 3 == 0 else 0.1 if (epoch // 20) % 3 == 1 else 1
    #sigma = 0 if epoch < 20 else 0.1 if epoch < 100 else 1

    # Save losses in history and update plot
    with torch.no_grad():
        history['imgs'] += [netG(fixed_noise).detach().cpu()]
        avgStart, avgEnd, valid = avg_start_end(history['imgs'][-1].detach(), start_max, end_max, start_min, end_min)

    history['lossD'] += [lossD.item()]
    history['lossG'] += [lossG.item()]
    history['valLossD'] += [valLossD.item()]
    history['avgStart'] += [avgStart.item()]
    history['avgEnd'] += [avgEnd.item()]
    history['valid'] += [valid.item()]
    history['validHist'] += [sum(history['valid'][-20:]) / len(history['valid'][-20:])]
    history['epoch'] += [epoch]

    viz.line(
        Y=Tensor(history['lossD']),
        X=Tensor(history['epoch']),
        win=winD,
        update='replace'
    )

    viz.line(
        Y=Tensor(history['lossG']),
        X=Tensor(history['epoch']),
        win=winG,
        update='replace'
    )

    viz.line(
        Y=Tensor(history['valid']),
        X=Tensor(history['epoch']),
        win=winValid,
        update='replace'
    )

    viz.line(
        Y=Tensor(history['validHist']),
        X=Tensor(history['epoch']),
        win=winValidHist,
        update='replace'
    )

    if epoch % 10 == 0:
        viz.images(
            history['imgs'][-1][:16].detach().round_().flip(2),
            opts=dict(title='Epoch' + str(epoch), width=1000, height=250)
        )

#for i, img in enumerate(history['imgs'][-1].detach().round_().flip(2)):
#    torchvision.utils.save_image(img, '../results/img/fake/' + str(i) + '.png')

#S = 0
#for data in loader['train']:
#    x_real = data['moves'].type(Tensor).flip(2)
#    for i, img in enumerate(x_real):
#        torchvision.utils.save_image(img, '../results/img/real/' + str(S + i) + '.png')
#    S += x_real.size(0)


# Save training output data into json file
import json

with open("../final_results/" + "nopenalties" + "_seed" + str(seed) + ".csv", "w") as history_file:
    json.dump(
        {k: history[k] for k in ('epoch', 'lossD', 'lossG', 'valLossD', 'valid', 'validHist')},
        history_file
    )
