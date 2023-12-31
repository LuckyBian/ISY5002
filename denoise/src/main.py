import argparse
import torch
import matplotlib.pyplot as plt
from data import NoisyBSDSDataset
from argument import parse
from model import UDnCNN
import nntools as nt
from utils import DenoisingStatsManager, plot

def run(args):
    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    # dataset
    train_set = NoisyBSDSDataset(
        args.root_dir, image_size=args.image_size, sigma=args.sigma)
    test_set = NoisyBSDSDataset(
        args.root_dir, mode='test', image_size=args.test_image_size, sigma=args.sigma)

    net = UDnCNN(args.D, C=args.C).to(device)

    # optimizer
    adam = torch.optim.Adam(net.parameters(), lr=args.lr)

    # stats manager
    stats_manager = DenoisingStatsManager()

    # experiment
    exp = nt.Experiment(net, train_set, test_set, adam, stats_manager, batch_size=args.batch_size,
                        output_dir=args.output_dir, perform_validation_during_training=True)

    # run
    if args.plot:
        fig, axes = plt.subplots(ncols=2, nrows=2, figsize=(9, 7))
        exp.run(num_epochs=args.num_epochs, plot=lambda exp: plot(exp, fig=fig, axes=axes, noisy=test_set[73][0]))
    else:
        exp.run(num_epochs=args.num_epochs)

    # Save the model
    torch.save(net.state_dict(), 'model.pth')
    print("Model saved as model.pth")

if __name__ == '__main__':
    args = parse()
    print(args)
    run(args)
