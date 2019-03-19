import argparse

parser = argparse.ArgumentParser(description='Convert a CUDA model to CPU.')
parser.add_argument("--infile", help="CUDA model filename")
parser.add_argument("--outfile", help="CPU model filename")
args = parser.parse_args()

net = torch.load(args.infile)
net = net.to('cpu')
torch.save(net, args.outfile)