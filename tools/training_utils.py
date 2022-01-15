import datetime
import torch
import matplotlib.pyplot as plt


class MonitorTraining:
    def __init__(self, x_train, y_train, x_val, y_val):
        self.date = str(datetime.datetime.now().date())
        self.curr_time = str(datetime.datetime.now().time())
        self.training_dirname = 'training/{}/{}'.format(self.date, self.curr_time)
        self.x_train = x_train
        self.y_train = y_train
        self.x_val = x_val
        self.y_val = y_val

    def plot_initial_data(self):
        for i in range(1):
            plt.clf()
            plt.plot(self.x_train[i])
            plt.plot(self.y_train[i])
            plt.savefig(self.training_dirname + '/training_trial.png')
            plt.clf()

        plt.plot(self.x_val[0])
        plt.plot(self.y_val[0])
        plt.plot(self.x_val[-1])
        plt.plot(self.y_val[-1])
        # plt.show()
        plt.savefig(self.training_dirname + '/validation_trial.png')

    def plot_progress(self, val_pred, lr):
        plt.clf()
        plt.plot(self.x_val[0])
        plt.plot(self.y_val[0])
        plt.plot(val_pred[0])
        plt.savefig(self.training_dirname + '/predicted_trial_lowest_{}.png'.format(lr))
        plt.clf()
        plt.plot(self.x_val[-1])
        plt.plot(self.y_val[-1])
        plt.plot(val_pred[-1])
        plt.savefig(self.training_dirname + '/predicted_trial_highest_{}.png'.format(lr))
        # loss_object = tf.keras.losses.MeanSquaredError(
        #     reduction=tf.keras.losses.Reduction.SUM_OVER_BATCH_SIZE)
        loss_object = None
        curr_loss = loss_object(self.y_val[:, :, 0], val_pred)
        if curr_loss <= 1e-6:
            return False
        return True

    def plot_loss(self, average_losses, lr):
        plt.clf()
        plt.plot(average_losses, label='train')
        # plt.ylim(bottom=0, top=0.001)
        plt.legend()
        plt.yscale("log")

#
def make_train_step(model, loss_fn, optimizer, clipping_value=None):
    # Builds function that performs a step in the train loop
    def train_step(x, y):
        # Sets model to TRAIN mode
        model.train()
        # Makes predictions
        output = model(x)
        # Computes loss
        loss = loss_fn(y, output.float())#output.view(-1).float())
        # Computes gradients
        loss.backward()
        if clipping_value is not None:
            torch.nn.utils.clip_grad_norm_(model.parameters(), clipping_value)

        # Updates parameters and zeroes gradients
        optimizer.step()
        optimizer.zero_grad()
        # Returns the loss
        return loss.item()

    # Returns the function that will be called inside the train loop
    return train_step


