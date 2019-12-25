import keras
import matplotlib.pyplot as plt


class LossHistory(keras.callbacks.Callback):
    def on_train_begin(self, logs={}):
        self.losses = {'batch': [], 'epoch': []}
        self.val_loss = {'batch': [], 'epoch': []}

    def on_batch_end(self, batch, logs={}):
        self.losses['batch'].append(logs.get('loss'))
        self.val_loss['batch'].append(logs.get('val_loss'))

    def on_epoch_end(self, batch, logs={}):
        self.losses['epoch'].append(logs.get('loss'))
        self.val_loss['epoch'].append(logs.get('val_loss'))

    def loss_plot(self, loss_type, savename):
        iters = range(len(self.losses[loss_type]))
        # 创建一个图
        plt.figure()

        # loss
        plt.plot(iters, self.losses[loss_type], 'g', label='train loss')
        if loss_type == 'epoch':
            # val_loss
            plt.plot(iters, self.val_loss[loss_type], 'k', label='val loss')
        plt.grid(True)  # 设置网格形式
        plt.xlabel(loss_type)
        plt.ylabel('loss')  # 给x，y轴加注释
        plt.legend(loc="upper right")  # 设置图例显示位置
        plt.savefig('./log/loss_{}_{}.png'.format(savename,loss_type))
        plt.show()
        
