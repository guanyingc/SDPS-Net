from collections import OrderedDict
import numpy as np

class Records(object):
    """
    Records->Train,Val->Loss,Accuracy->Epoch1,2,3->[v1,v2]
    IterRecords->Train,Val->Loss, Accuracy,->[v1,v2]
    """
    def __init__(self, log_dir, records=None):
        if records == None:
            self.records = OrderedDict()
        else:
            self.records = records
        self.iter_rec = OrderedDict()
        self.log_dir  = log_dir
        self.classes = ['loss', 'acc', 'err', 'ratio']

    def resetIter(self):
        self.iter_rec.clear()

    def checkDict(self, a_dict, key, sub_type='dict'):
        if key not in a_dict.keys():
            if sub_type == 'dict':
                a_dict[key] = OrderedDict()
            if sub_type == 'list':
                a_dict[key] = []

    def updateIter(self, split, keys, values):
        self.checkDict(self.iter_rec, split, 'dict')
        for k, v in zip(keys, values):
            self.checkDict(self.iter_rec[split], k, 'list')
            self.iter_rec[split][k].append(v)

    def saveIterRecord(self, epoch, reset=True):
        for s in self.iter_rec.keys(): # s stands for split
            self.checkDict(self.records, s, 'dict')
            for k in self.iter_rec[s].keys():
                self.checkDict(self.records[s], k, 'dict')
                self.checkDict(self.records[s][k], epoch, 'list')
                self.records[s][k][epoch].append(np.mean(self.iter_rec[s][k]))
        if reset: 
            self.resetIter()

    def insertRecord(self, split, key, epoch, value):
        self.checkDict(self.records, split, 'dict')
        self.checkDict(self.records[split], key, 'dict')
        self.checkDict(self.records[split][key], epoch, 'list')
        self.records[split][key][epoch].append(value)

    def iterRecToString(self, split, epoch):
        rec_strs = ''
        for c in self.classes:
            strs = ''
            for k in self.iter_rec[split].keys():
                if (c in k.lower()):
                    strs += '{}: {:.3f}| '.format(k, np.mean(self.iter_rec[split][k]))
            if strs != '':
                rec_strs += '\t [{}] {}\n'.format(c.upper(), strs)
        self.saveIterRecord(epoch)
        return rec_strs

    def epochRecToString(self, split, epoch):
        rec_strs = ''
        for c in self.classes:
            strs = ''
            for k in self.records[split].keys():
                if (c in k.lower()) and (epoch in self.records[split][k].keys()):
                    strs += '{}: {:.3f}| '.format(k, np.mean(self.records[split][k][epoch]))
            if strs != '':
                rec_strs += '\t [{}] {}\n'.format(c.upper(), strs)
        return rec_strs

    def recordToDictOfArray(self, splits, epoch=-1, intv=1):
        if len(self.records) == 0: return {}
        if type(splits) == str: splits = [splits]

        dict_of_array = OrderedDict()
        for split in splits:
            for k in self.records[split].keys():
                y_array, x_array = [], []
                if epoch < 0:
                    for ep in self.records[split][k].keys():
                        y_array.append(np.mean(self.records[split][k][ep]))
                        x_array.append(ep)
                else:
                    if epoch in self.records[split][k].keys():
                        y_array = np.array(self.records[split][k][epoch])
                        x_array = np.linspace(intv, intv*len(y_array), len(y_array))
                dict_of_array[split[0] + split[-1] + '_' + k]      = y_array
                dict_of_array[split[0] + split[-1] + '_' + k+'_x'] = x_array
        return dict_of_array
