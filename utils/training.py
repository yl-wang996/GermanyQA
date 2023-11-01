import torch
import os
class Earlystopping():
    def __init__(self,mode,patience):
        self.best = 0
        self.mode = mode
        self.patience = patience
        self.counter = 0
        assert mode in ['max','min'], 'Please check your mode, option from "max" or "min"'

    def _check_best(self, value):
        if self.mode=='max' and value>self.best:
            return True
        elif self.mode=='min' and value<self.best:
            return True
        return False

    def update(self,value):
        if self._check_best(value):
            self.best = value
            self.counter = 0
            return True
        else:
            self.counter += 1
            return False

    def stop(self):
        if self.counter > self.patience:
            return True
        else:
            self.counter = 0
            return False

def model_save(model, epoch, recorder, config, max_num=5):
    model_name = f'Ep{epoch}_ValF1_{recorder.val_recoder.f1_avg:.4f}_TestF1_{recorder.test_recoder.f1_avg:.4f}.pth'
    torch.save(model.state_dict(), os.path.join(config.model_dir, model_name))
    folders = os.listdir(config.model_dir)
    name_f1_map = {}
    if len(folders)>max_num:
        for folder in  folders:
            val_f1 = float(folder[:-4].split('_')[2])
            name_f1_map[val_f1]= folder
        min_val = min(list(name_f1_map.keys()))
        min_folder = name_f1_map[min_val]
        os.remove(os.path.join(config.model_dir,min_folder))
