from models import model_utils
from utils  import eval_utils, time_utils

def train(args, loader, model, criterion, optimizer, log, epoch, recorder):
    model.train()
    log.printWrite('---- Start Training Epoch %d: %d batches ----' % (epoch, len(loader)))
    timer = time_utils.Timer(args.time_sync);

    for i, sample in enumerate(loader):
        data = model_utils.parseData(args, sample, timer, 'train')
        input = model_utils.getInput(args, data)

        pred = model(input); timer.updateTime('Forward')

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
            results, recorder, nrow = prepareSave(args, data, pred, recorder, log) 
            log.saveImgResults(results, 'train', epoch, iters, nrow=nrow)
            log.plotCurves(recorder, 'train', epoch=epoch, intv=args.train_disp)

        if args.max_train_iter > 0 and iters >= args.max_train_iter: break
    opt = {'split': 'train', 'epoch': epoch, 'recorder': recorder}
    log.printEpochSummary(opt)

def prepareSave(args, data, pred, recorder, log):
    results = [data['img'].data, data['m'].data, (data['n'].data+1)/2]
    if args.s1_est_d:
        l_acc, data['dir_err'] = eval_utils.calDirsAcc(data['dirs'].data, pred['dirs'].data, args.batch)
        recorder.updateIter('train', l_acc.keys(), l_acc.values())

    if args.s1_est_i:
        int_acc, data['int_err'] = eval_utils.calIntsAcc(data['ints'].data, pred['intens'].data, args.batch)
        recorder.updateIter('train', int_acc.keys(), int_acc.values())

    if args.s1_est_n:
        acc, error_map = eval_utils.calNormalAcc(data['n'].data, pred['n'].data, data['m'].data)
        pred_n = (pred['n'].data + 1) / 2
        masked_pred = pred_n * data['m'].data.expand_as(pred['n'].data)
        res_n = [pred_n, masked_pred, error_map['angular_map']]
        results += res_n
        recorder.updateIter('train', acc.keys(), acc.values())
    nrow = data['img'].shape[0] if data['img'].shape[0] <= 32 else 32
    return results, recorder, nrow
