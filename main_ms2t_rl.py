import os
import numpy
from time import time
from parser_ms2t import get_args
from chem_lib.utils import count_model_params
import sys
from torch.utils.tensorboard import SummaryWriter
import torch
from chem_lib.models import cd_maml, ContextAwareRelationNet
import pickle


def main():
    root_dir = '.'
    args = get_args(root_dir)
    model = ContextAwareRelationNet(args)
    count_model_params(model)
    model = model.to(args.device)

    trainer = cd_maml(args, model)

    writer = SummaryWriter()
    best_avg_auc = 0
    moving_avg = 0

    for epoch in range(1, args.epochs + 1):
        _, moving_avg = trainer.train_step(writer, epoch, moving_avg)

        if epoch % args.eval_steps == 0 or epoch == 1 or epoch == args.epochs:
            avg = []
            for _ in range(1):
                best_avg_auc = trainer.test_step(writer, epoch)

                avg.append(best_avg_auc)
        if epoch % args.save_steps == 0:
            trainer.save_model(epoch)

    print('Train done.')
    print('Best Avg AUC:', best_avg_auc)
    print('Train conclusion:')
    trainer.conclude()

    if args.save_logs:
        trainer.save_result_log()


if __name__ == "__main__":
    main()
