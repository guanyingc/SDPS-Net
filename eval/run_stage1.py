import torch, sys
sys.path.append('.')

from datasets import custom_data_loader
from options  import run_model_opts
from models   import custom_model
from utils    import logger, recorders

import test_stage1 as test_utils

args = run_model_opts.RunModelOpts().parse()
log  = logger.Logger(args)

def main(args):
    test_loader = custom_data_loader.benchmarkLoader(args)
    model    = custom_model.buildModel(args)
    recorder = recorders.Records(args.log_dir)
    test_utils.test(args, 'test', test_loader, model, log, 1, recorder)
    log.plotCurves(recorder, 'test')

if __name__ == '__main__':
    torch.manual_seed(args.seed)
    main(args)
