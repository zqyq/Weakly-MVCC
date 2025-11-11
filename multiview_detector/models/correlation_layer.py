import torch
import torch.nn as nn
import torch.nn.functional as F


def _corr_cal(A, B):
    b, n, c, h, w = A.shape

    # norm_A = torch.pow(torch.sum(torch.pow(A, 2), dim=2, keepdim=True), 0.5)
    # norm_B = torch.pow(torch.sum(torch.pow(B, 2), dim=2, keepdim=True), 0.5)
    # feature_A = torch.divide(A, norm_A + 1e-8)
    # feature_B = torch.divide(B, norm_B + 1e-8)

    # A_flatten = torch.reshape(feature_A, [b, n, c, h*w])
    # B_flatten = torch.reshape(feature_B, [b, n, c, h*w]).permute(0, 1, 3, 2)

    # corr_AB = torch.matmul(B_flatten, A_flatten)
    # output1 = []
    # for i in range(b):
    #     corr_AB_i = torch.bmm(B_flatten[i], A_flatten[i])
    #     output1.append(corr_AB_i)
    # output1 = torch.stack(output1, dim=0)
    # output1 = torch.reshape(output1, [b, n, h, w, h*w]).permute(0, 1, 4, 2, 3)

    A_flatten = torch.reshape(A, [b * n, c, h * w])
    B_flatten = torch.reshape(B, [b * n, c, h * w]).permute(0, 2, 1)
    corr_AB = torch.bmm(B_flatten, A_flatten)
    output = torch.reshape(corr_AB, [b, n, h, w, h * w]).permute(0, 1, 4, 2, 3)

    return output

class Correlation_Layer(nn.Module):
    def __init__(self, view_size):
        self.view_size = view_size
        super().__init__()


    def forward(self, x):
        n = self.view_size
        b, c, h, w = x.shape
        b = int(b / n)

        x = F.avg_pool2d(x, kernel_size=(2, 2), stride=(2, 2), padding=0)
        x = torch.reshape(x, (b, n, c, int(h/2), int(w/2)))

        # corr = torch.zeros((1, n-1, int(h*w/4), int(h/2), int(w/2))).to(x.device)
        corr = []
        feature_B = x
        for i in range(n):
            feature_A_i = x[:, i:i+1]
            feature_A = feature_A_i.repeat(1, n, 1, 1, 1)
            corr_i = _corr_cal(feature_A, feature_B)

            # delete self correlation map
            corr_i = torch.cat([corr_i[:, :i], corr_i[:, i+1:]], dim=1)
            corr.append(corr_i)

        corr = torch.cat(corr, dim=1)
        corr = corr.reshape((-1, int(h/2)*int(w/2), int(h/2), int(w/2)))

        return corr

class Correlation_Layer_noNorm(nn.Module):
    def __init__(self, view_size):
        self.view_size = view_size
        super().__init__()

    def forward(self, x):
        n = self.view_size
        b, c, h, w = x.shape
        b = int(b / n)

        norm_feature = torch.pow(torch.sum(torch.pow(x, 2), dim=1, keepdim=True), 0.5)
        feature_norm = torch.divide(x, norm_feature+1e-8)

        feature_norm = torch.reshape(feature_norm, (b, n, c, h, w))

        corr = []
        feature_A = feature_norm
        for i in range(n):
            feature_B_i = feature_norm[:, i:i+1]
            feature_B = feature_B_i.repeat(1, n, 1, 1, 1)

            corr_i = _corr_cal(feature_A, feature_B)
            # 顺序为从前往后顺序
            corr_i = torch.cat([corr_i[:, :i], corr_i[:, i + 1:]], dim=1)
            corr.append(corr_i)

        corr = torch.cat(corr, dim=1)
        corr = corr.reshape((-1, h*w, h, w))

        return corr



if __name__ == '__main__':
    # x = torch.asarray([[[[1, 2, 3, 2], [4, 5, 6, 1], [7, 8, 9, 0], [3, 8, 9, 2]]]])
    # y = torch.asarray([[[[4, 0, 7, 1], [3, 8, 9, 2], [2, 6, 1, 6], [1, 2, 3, 2]]]])
    # z = torch.asarray([[[[7, 3, 6, 8], [6, 3, 5, 1], [9, 4, 3, 8], [4, 2, 8, 7]]]])
    corr_Layer = Correlation_Layer(5)
    # intput = torch.cat([x, y, z], dim=0)
    intput = torch.ones((10, 3, 4, 4))
    intput[5:] = intput[5:] * 2
    output = corr_Layer(intput)
    print(output.shape)