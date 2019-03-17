import torch
from models import model_utils
from utils  import eval_utils, time_utils

def train(args, loader, models, criterion, optimizers, log, epoch, recorder):
    models[1].train()
    models[0].eval()
    optimizer, optimizer_c = optimizers
    log.printWrite('---- Start Training Epoch %d: %d batches ----' % (epoch, len(loader)))
    timer = time_utils.Timer(args.time_sync);

    for i, sample in enumerate(loader):
        data = model_utils.parseData(args, sample, timer, 'train')
        input = model_utils.getInput(args, data)
        with torch.no_grad():
            pred_c = models[0](input); 
        input.append(pred_c)
        pred = models[1](input); timer.updateTime('Forward')

        optimizer.zero_grad()

        loss = criterion.forward(pred, data); 
        timer.updateTime('Crit');
        criterion.backward(); timer.updateTime('Backward')

        recorder.updateIter('train', loss.keys(), loss.values())

        optimizer.step(); timer.updateTime('Solver')

        iters = i + 1
        if iters % args.train_disp == 0:
            opt = {'split':'train', 'epoch':epoch, 'iters':iters, 'batch':len(loader), 
                    'timer':timer, 'recorder': recorder}
            log.printItersSummary(opt)

        if iters % args.train_save == 0:
            results, recorder, nrow = prepareSave(args, data, pred_c, pred, recorder, log) 
            log.saveImgResults(results, 'train', epoch, iters, nrow=nrow)
            log.plotCurves(recorder, 'train', epoch=epoch, intv=args.train_disp)

        if args.max_train_iter > 0 and iters >= args.max_train_iter: break
    opt = {'split': 'train', 'epoch': epoch, 'recorder': recorder}
    log.printEpochSummary(opt)

def prepareSave(args, data, pred_c, pred, recorder, log):
    input_var, mask_var = data['img'], data['m']
    results = [input_var.data, mask_var.data, (data['n'].data+1)/2]
    if args.s1_est_d:
        l_acc, data['dir_err'] = eval_utils.calDirsAcc(data['dirs'].data, pred_c['dirs'].data, args.batch)
        recorder.updateIter('train', l_acc.keys(), l_acc.values())
    if args.s1_est_i:
        int_acc, data['int_err'] = eval_utils.calIntsAcc(data['ints'].data, pred_c['intens'].data, args.batch)
        recorder.updateIter('train', int_acc.keys(), int_acc.values())

    if args.s2_est_n:
        acc, error_map = eval_utils.calNormalAcc(data['n'].data, pred['n'].data, mask_var.data)
        pred_n = (pred['n'].data + 1) / 2
        masked_pred = pred_n * mask_var.data.expand_as(pred['n'].data)
        res_n = [masked_pred, error_map['angular_map']]
        results += res_n
        recorder.updateIter('train', acc.keys(), acc.values())

    nrow = input_var.shape[0] if input_var.shape[0] <= 32 else 32
    return results, recorder, nrow
