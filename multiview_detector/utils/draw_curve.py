# import matplotlib
#
# matplotlib.use('agg')
import matplotlib.pyplot as plt


def draw_curve(path, x_epoch, train_loss, train_prec, test_loss, test_prec, test_moda=None):
    fig = plt.figure()
    ax1 = fig.add_subplot(131, title="loss")
    ax2 = fig.add_subplot(132, title="prec_mae")
    ax1.plot(x_epoch, train_loss, 'bo-', label='train' + ': {:.3f}'.format(train_loss[-1]))
    ax1.plot(x_epoch, test_loss, 'ro-', label='test' + ': {:.3f}'.format(test_loss[-1]))
    ax2.plot(x_epoch, train_prec, 'bo-', label='train' + ': {:.1f}'.format(train_prec[-1]))
    ax2.plot(x_epoch, test_prec, 'ro-', label='test' + ': {:.1f}'.format(test_prec[-1]))

    ax1.legend()
    ax2.legend()
    if test_moda is not None:
        ax3 = fig.add_subplot(133, title="moda")
        ax3.plot(x_epoch, test_moda, 'ro-', label='test' + ': {:.1f}'.format(test_moda[-1]))
        ax3.legend()
    fig.savefig(path)
    plt.close(fig)


    # fig = plt.figure()
    # ax1 = fig.add_subplot(121, title="train_loss")
    # ax2 = fig.add_subplot(122, title="test_loss")
    # ax1.plot(x_epoch, train_loss, 'bo-', label='train' + ': {:.3f}'.format(train_loss[-1]))
    # ax2.plot(x_epoch, test_loss, 'ro-', label='test' + ': {:.3f}'.format(test_loss[-1]))
    # # ax2.plot(x_epoch, train_prec, 'bo-', label='train' + ': {:.1f}'.format(train_prec[-1]))
    # # ax2.plot(x_epoch, test_prec, 'ro-', label='test' + ': {:.1f}'.format(test_prec[-1]))
    #
    # ax1.legend()
    # ax2.legend()
    # # if test_moda is not None:
    # #     ax3 = fig.add_subplot(133, title="moda")
    # #     ax3.plot(x_epoch, test_moda, 'ro-', label='test' + ': {:.1f}'.format(test_moda[-1]))
    # #     ax3.legend()
    # fig.savefig(path)
    # plt.close(fig)

if __name__ == "__main__":
    path = '/root/home/Daijie/Semi_2D_Counting/logs/wildtrack_frame/1.jpg'
    epoch = [0, 1, 2, 3, 4, 5]
    train_loss = [1.5, 0.708, 0.845, 0.505, 0.49, 0.426]
    test_loss = [0.0, 0.0, 0., 0., 0., 0.]
    fig = plt.figure()
    ax1 = fig.add_subplot(121, title='train_loss')
    ax2 = fig.add_subplot(122, title='test_loss')
    ax1.plot(epoch, train_loss, 'bo-', label='train' + ': {:.3f}'.format(train_loss[-1]))
    ax2.plot(epoch, test_loss, 'ro-', label='test' + ': {:.3f}'.format(test_loss[-1]))

    fig.savefig(path)
    plt.close(fig)