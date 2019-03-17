import torch
from options  import stage2_opts
from utils    import logger, recorders
from datasets import custom_data_loader
from models   import custom_model, solver_utils, model_utils

import train_stage2 as train_utils
import test_stage2 as test_utils

args = stage2_opts.TrainOpts().parse()
log  = logger.Logger(args)

def main(args):
    model = custom_model.buildModel(args)
    model_s2 = custom_model.buildModelStage2(args)
    models = [model, model_s2]

    optimizer, scheduler, records = solver_utils.configOptimizer(args, model_s2)
    optimizers = [optimizer, -1]
    criterion = solver_utils.Stage2Crit(args)
    recorder  = recorders.Records(args.log_dir, records)

    train_loader, val_loader = custom_data_loader.customDataloader(args)

    for epoch in range(args.start_epoch, args.epochs+1):
        scheduler.step()

        recorder.insertRecord('train', 'lr', epoch, scheduler.get_lr()[0])

        train_utils.train(args, train_loader, models, criterion, optimizers, log, epoch, recorder)
        if epoch % args.save_intv == 0: 
            model_utils.saveCheckpoint(args.cp_dir, epoch, model_s2, optimizer, recorder.records, args)
        log.plotCurves(recorder, 'train')

        if epoch % args.val_intv == 0:
            test_utils.test(args, 'val', val_loader, models, log, epoch, recorder)
            log.plotCurves(recorder, 'val')

if __name__ == '__main__':
    torch.manual_seed(args.seed)
    main(args)
