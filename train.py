import tensorflow as tf
from network.networks import fit_network
from network.ice_loader import train_test
from network.parser import Parser

def main():
    args = Parser().get_parser().parse_args()
    dir = args.data_dir
    train_X, train_y, test_X, test_y = train_test(dir,args.n_steps_in,args.n_steps_out)
    fit_network(train_X, train_y, test_X, test_y, args)

if  __name__ == '__main__' :
    main()