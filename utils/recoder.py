from torch.utils.tensorboard import SummaryWriter

class SingleRecorder():
    def __init__(self, display_freq=20,prefix=None):
        self.em_sum = 0
        self.loss_sum = 0
        self.f1_sum = 0
        self.precision_sum = 0
        self.recall_sum = 0
        self.step = 0
        self.em_avg = 0
        self.loss_avg = 0
        self.f1_avg = 0
        self.precision_avg = 0
        self.recall_avg = 0
        self.display_freq = display_freq
        self.prefix = prefix

    def refresh(self):
        self.em_sum = 0
        self.loss_sum = 0
        self.f1_sum = 0
        self.precision_sum = 0
        self.recall_sum = 0
        self.step = 0

    def update(self,loss,em,f1,precision, recall):
        self.loss_sum += loss
        self.em_sum += em
        self.f1_sum += f1
        self.precision_sum += precision
        self.recall_sum += recall
        self.step += 1
        self.loss_avg = self.loss_sum/self.step
        self.em_avg = self.em_sum/self.step
        self.f1_avg = self.f1_sum/self.step
        self.precision_avg = self.precision_sum/self.step
        self.recall_avg = self.recall_sum/self.step

    def display(self,i,step_per_epoch):

        if i%self.display_freq == 0 or i== step_per_epoch:
            print(f'{self.prefix} Step[{i}/{step_per_epoch}]    '
                  f'{self.prefix}_loss:{self.loss_avg:.4f}    '
                  f'{self.prefix}_em:{self.em_avg:.4f}    '
                  f'{self.prefix}_f1:{self.f1_avg:.4f}    '
                  f'{self.prefix}_precision:{self.precision_avg:.4f}    '
                  f'{self.prefix}_recall:{self.recall_avg:.4f}')

class Recorder():
    def __init__(self,display_freq=20, logdir='./tensorboard_log'):
        self.display_freq = display_freq
        self.tr_recoder = SingleRecorder(display_freq=display_freq,
                                         prefix='train')
        self.val_recoder = SingleRecorder(display_freq=display_freq,
                                         prefix='val')
        self.test_recoder = SingleRecorder(display_freq=display_freq,
                                         prefix='test')
        self.writer = SummaryWriter(logdir)

    def update(self,loss, em, f1, precision, recall, prefix='train'):
        if prefix=='train':
            self.tr_recoder.update(loss, em, f1, precision, recall)
        elif prefix=='val':
            self.val_recoder.update(loss, em, f1, precision, recall)
        elif prefix=='test':
            self.test_recoder.update(loss, em, f1, precision, recall)
        else:
            assert False, 'Please input the \'prefix\' from [ train / val / test ].'

    def record2log(self,epoch,prefix,recorder):
        self.writer.add_scalar(f'loss/{prefix}_loss', recorder.loss_avg, epoch)
        self.writer.add_scalar(f'em/{prefix}_em', recorder.em_avg, epoch)
        self.writer.add_scalar(f'f1/{prefix}_f1', recorder.f1_avg, epoch)
        self.writer.add_scalar(f'precison/{prefix}_precison', recorder.precision_avg, epoch)
        self.writer.add_scalar(f'recall/{prefix}_recall', recorder.recall_avg, epoch)


    def display(self,i,step_per_epoch,prefix='train'):
        if prefix=='train':
            self.tr_recoder.display(i,step_per_epoch)
        elif prefix=='val':
            self.val_recoder.display(i,step_per_epoch)
        elif prefix=='test':
            self.test_recoder.display(i,step_per_epoch)
        else:
            assert False, 'Please input the \'prefix\' from [ train / val / test ].'