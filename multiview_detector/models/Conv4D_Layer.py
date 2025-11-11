import torch
import torch.nn as nn
import torch.nn.functional as F


class Conv4D(nn.Module):
    def __init__(self, filters, kernel_size, padding='same', activation='relu'):
        self.filters = filters
        self.kernel_size = kernel_size #must be a tuple!!!!
        self.padding=padding
        self.activation = activation

        super(Conv4D, self).__init__()

    def build(self, input_shape):
        spatialDims = len(self.kernel_size)
        allDims = len(input_shape)
        assert allDims == spatialDims + 2  # spatial dimensions + batch size + channels

        kernelShape = self.kernel_size + (input_shape[-1], self.filters)
        biasShape = tuple(1 for _ in range(allDims - 1)) + (self.filters,)

        # 初始化卷积核权重
        self.kernel = nn.Parameter(torch.Tensor(*kernelShape))
        nn.init.uniform_(self.kernel)

        # 初始化偏置
        self.bias = nn.Parameter(torch.zeros(*biasShape))

    def conv4d(self, input, filters, kernel_size, kernel, strides=(1, 1, 1, 1), padding='same', dilation_rate=(1, 1, 1, 1), reuse=None):
        assert len(input.get_shape.as_list()) == 6, ("Tensor of shape (b, c, l, d, h, w) expected")
        assert len(kernel_size) == 4, "4D kernel size expected"
        assert strides == (1, 1, 1, 1), "Strides other than 1 not yet implemented"
        assert dilation_rate == (1, 1, 1, 1), "Dilation rate other than 1 not yet implemented"

        (b, l_i, d_i, h_i, w_i, c_i) = tuple(input.get_shape.as_list())
        (l_k, d_k, h_k, w_k) = kernel_size

        if padding == 'valid':
            (l_o, d_o, h_o, w_o) = (
                l_i - l_k + 1,
                d_i - d_k + 1,
                h_i - h_k + 1,
                w_i - w_k + 1
            )
        else:
            (l_o, d_o, h_o, w_o) = (l_i, d_i, h_i, w_i)

        # output tensors for each 3D frame
        frame_results = [None] * l_o

        # convolve each kernel frame i with each input frame j
        for i in range(l_k):

            # reuse variables of previous 3D convolutions for the same kernel
            # frame (or if the user indicated to have all variables reused)
            reuse_kernel = reuse

            for j in range(l_i):

                # add results to this output frame
                out_frame = j - (i - l_k // 2) - (l_i - l_o) // 2
                if out_frame < 0 or out_frame >= l_o:
                    continue

                # convolve input frame j with kernel frame i
                input_ij = torch.reshape(input[:, j], (b, c_i, d_i, h_i, w_i))
                # kernel_i = kernel[i, ...]
                kernel_i = self.kernel_size[i]
                frame_conv3d = nn.Conv3d(c_i, self.filters, kernel_i, stride=strides, padding=padding, dilation=dilation_rate)(input_ij)
                reuse_kernel = True

                if frame_results[out_frame] is None:
                    frame_results[out_frame] = frame_conv3d
                else:
                    frame_results[out_frame] += frame_conv3d

        output = torch.stack(frame_results, dim=1)

        # if activation:
        #     output = activation(output)
        return output


    def forward(self, x):
        results = self.conv4d(x, 1, self.kernel_size, self.kernel, padding=self.padding)

        if self.activation == 'ReLU' or self.activation == 'relu':
            results = torch.relu(results + self.bias)

        if self.activation == 'SIGMOID' or self.activation == 'sigmoid':
            results = torch.sigmoid(results + self.bias)

        return results





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