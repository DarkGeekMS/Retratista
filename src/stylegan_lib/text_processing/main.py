import argparse
import os
from trainer import Trainer

def main():
    # arguments parsing
    argparser = argparse.ArgumentParser(description=__doc__)
    argparser.add_argument('-dataset_path', '--path_to_dataset', type=str, help='path to dataset folder that contains folder for each label', default = '../dataset/')
    argparser.add_argument('-arch', '--architecture', type=str, help='model architecture, choose from [bert-base-uncased, distilbert-base-uncased, albert-base-v2, roberta-base], default is distilbert-base-uncased!', default = 'distilbert-base-uncased')
    argparser.add_argument('-lr', '--learning_rate', type=float, help='starting learning rate', default=0.001)
    argparser.add_argument('-lr_decay', '--learning_rate_decay', type=float, help='decay rate of learning rate after step size', default=0.2)
    argparser.add_argument('-lr_decay_step_size', '--learning_rate_decay_step_size', type=int, help='number of epochs after which decay learning rate occurs', default=5)
    argparser.add_argument('-batch', '--batch_size', type=int, help='batch size', default=32)
    argparser.add_argument('-validate_step', '--validate_step', type=int, help='validate each how many epochs', default=1)
    argparser.add_argument('-epochs', '--num_of_epochs', type=int, help='num of epochs', default=50)
    argparser.add_argument("--resume_from_last_trial", action='store_true')
    args = argparser.parse_args()

    if args.architecture != 'distilbert-base-uncased' and args.architecture != 'albert-base-v2' and args.architecture != 'roberta-base' and args.architecture != 'bert-base-uncased':
        print('incorrect model architecture! choose from [bert-base-uncased, distilbert-base-uncased, albert-base-v2, roberta-base].')
        exit()
    
    configs = {
        'lr': args.learning_rate,
        'lr_decay': args.learning_rate_decay,
        'lr_decay_step_size': args.learning_rate_decay_step_size,
        'batch_size': args.batch_size,
        'num_of_epochs': args.num_of_epochs,
        'architecture': args.architecture,
        'resume': args.resume_from_last_trial     
    }
    trainer = Trainer(configs)
    trainer.train(args.validate_step)


if __name__ == '__main__':
    main()
        
        
                                                                
                            



 

