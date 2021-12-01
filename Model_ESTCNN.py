import torch
import torch.nn as nn
import random

class ESTCNN(nn.Module):
    def __init__(self, n_classes, input_ch, input_time, batch_norm_alpha=0.1):
        super(ESTCNN, self).__init__()
        self.batch_norm_alpha = batch_norm_alpha
        self.n_classes = n_classes
        n_ch1, n_ch2, n_ch3 = 16, 32, 64

        self.convnet = nn.Sequential(
            nn.Conv2d(1, n_ch1, kernel_size=(1, 3), stride=1, padding="valid"),
            nn.ReLU(),
            nn.BatchNorm2d(n_ch1, momentum=self.batch_norm_alpha, affine=True, eps=1e-5),
            nn.Conv2d(n_ch1, n_ch1, kernel_size=(1, 3), stride=1, padding="valid"),
            nn.ReLU(),
            nn.BatchNorm2d(n_ch1, momentum=self.batch_norm_alpha, affine=True, eps=1e-5),
            nn.Conv2d(n_ch1, n_ch1, kernel_size=(1, 3), stride=1, padding="valid"),
            nn.ReLU(),
            nn.BatchNorm2d(n_ch1, momentum=self.batch_norm_alpha, affine=True, eps=1e-5),
            nn.MaxPool2d(kernel_size=(1, 2), stride=(1, 2)),

            nn.Conv2d(n_ch1, n_ch2, kernel_size=(1, 3), stride=1, padding="valid"),
            nn.ReLU(),
            nn.BatchNorm2d(n_ch2, momentum=self.batch_norm_alpha, affine=True, eps=1e-5),
            nn.Conv2d(n_ch2, n_ch2, kernel_size=(1, 3), stride=1, padding="valid"),
            nn.ReLU(),
            nn.BatchNorm2d(n_ch2, momentum=self.batch_norm_alpha, affine=True, eps=1e-5),
            nn.Conv2d(n_ch2, n_ch2, kernel_size=(1, 3), stride=1, padding="valid"),
            nn.ReLU(),
            nn.BatchNorm2d(n_ch2, momentum=self.batch_norm_alpha, affine=True, eps=1e-5),
            nn.MaxPool2d(kernel_size=(1, 2), stride=(1, 2)),

            nn.Conv2d(n_ch2, n_ch3, kernel_size=(1, 3), stride=1, padding="valid"),
            nn.ReLU(),
            nn.BatchNorm2d(n_ch3, momentum=self.batch_norm_alpha, affine=True, eps=1e-5),
            nn.Conv2d(n_ch3, n_ch3, kernel_size=(1, 3), stride=1, padding="valid"),
            nn.ReLU(),
            nn.BatchNorm2d(n_ch3, momentum=self.batch_norm_alpha, affine=True, eps=1e-5),
            nn.Conv2d(n_ch3, n_ch3, kernel_size=(1, 3), stride=1, padding="valid"),
            nn.ReLU(),
            nn.BatchNorm2d(n_ch3, momentum=self.batch_norm_alpha, affine=True, eps=1e-5),
            nn.AvgPool2d(kernel_size=(1, 7), stride=(1, 7)),
        )
        self.convnet.eval()
        out = self.convnet(torch.zeros(1, 1, input_ch, input_time))

        n_out_time = out.cpu().data.numpy().shape[3]
        self.final_conv_length = n_out_time

        self.n_outputs = out.size()[1] * out.size()[2] * out.size()[3]

        """ Classifier """
        self.clf = nn.Sequential(nn.Linear(self.n_outputs, 50),
                                 nn.ReLU(),
                                 nn.Linear(50, self.n_classes),
                                 nn.Sigmoid()
                                 )
    def forward(self, x):
        output = self.convnet(x)
        output = output.view(output.size()[0], -1)
        output = self.clf(output)
        return output

    def get_embedding(self, x):
        return self.forward(x)

class Proposed_model(nn.Module):
    def __init__(self, n_classes, input_ch, input_time, batch_norm_alpha=0.1):
        super(Proposed_model, self).__init__()
        self.batch_norm_alpha = batch_norm_alpha
        self.n_classes = n_classes
        n_ch1, n_ch2, n_ch3, n_ch4 = 16, 32, 64, 128

        self.convnet = nn.Sequential(
            nn.Conv2d(1, n_ch1, kernel_size=(1, 3), stride=1, padding="valid"),
            nn.ReLU(),
            nn.BatchNorm2d(n_ch1, momentum=self.batch_norm_alpha, affine=True, eps=1e-5),
            nn.Dropout(p=0.2),
            nn.Conv2d(n_ch1, n_ch1, kernel_size=(1, 5), stride=1, padding="valid"),
            nn.ReLU(),
            nn.BatchNorm2d(n_ch1, momentum=self.batch_norm_alpha, affine=True, eps=1e-5),
            nn.Dropout(p=0.2),
            nn.Conv2d(n_ch1, n_ch1, kernel_size=(1, 3), stride=1, padding="valid"),
            nn.ReLU(),
            nn.BatchNorm2d(n_ch1, momentum=self.batch_norm_alpha, affine=True, eps=1e-5),
            nn.Dropout(p=0.2),
            nn.MaxPool2d(kernel_size=(1, 2), stride=(1, 2)),

            nn.Conv2d(n_ch1, n_ch2, kernel_size=(1, 3), stride=1, padding="valid"),
            nn.ReLU(),
            nn.BatchNorm2d(n_ch2, momentum=self.batch_norm_alpha, affine=True, eps=1e-5),
            nn.Dropout(p=0.2),
            nn.Conv2d(n_ch2, n_ch2, kernel_size=(1, 5), stride=1, padding="valid"),
            nn.ReLU(),
            nn.BatchNorm2d(n_ch2, momentum=self.batch_norm_alpha, affine=True, eps=1e-5),
            nn.Dropout(p=0.2),
            nn.Conv2d(n_ch2, n_ch2, kernel_size=(1, 3), stride=1, padding="valid"),
            nn.ReLU(),
            nn.BatchNorm2d(n_ch2, momentum=self.batch_norm_alpha, affine=True, eps=1e-5),
            nn.Dropout(p=0.2),
            nn.MaxPool2d(kernel_size=(1, 2), stride=(1, 2)),

            nn.Conv2d(n_ch2, n_ch3, kernel_size=(1, 3), stride=1, padding="valid"),
            nn.ReLU(),
            nn.BatchNorm2d(n_ch3, momentum=self.batch_norm_alpha, affine=True, eps=1e-5),
            nn.Dropout(p=0.2),
            nn.Conv2d(n_ch3, n_ch3, kernel_size=(1, 5), stride=1, padding="valid"),
            nn.ReLU(),
            nn.BatchNorm2d(n_ch3, momentum=self.batch_norm_alpha, affine=True, eps=1e-5),
            nn.Dropout(p=0.2),
            nn.Conv2d(n_ch3, n_ch3, kernel_size=(1, 3), stride=1, padding="valid"),
            nn.ReLU(),
            nn.BatchNorm2d(n_ch3, momentum=self.batch_norm_alpha, affine=True, eps=1e-5),
            nn.Dropout(p=0.2),
            nn.MaxPool2d(kernel_size=(1, 2), stride=(1, 2)),

            nn.Conv2d(n_ch3, n_ch4, kernel_size=(1, 3), stride=1, padding="valid"),
            nn.ReLU(),
            nn.BatchNorm2d(n_ch4, momentum=self.batch_norm_alpha, affine=True, eps=1e-5),
            nn.Dropout(p=0.2),
            nn.Conv2d(n_ch4, n_ch4, kernel_size=(1, 5), stride=1, padding="valid"),
            nn.ReLU(),
            nn.BatchNorm2d(n_ch4, momentum=self.batch_norm_alpha, affine=True, eps=1e-5),
            nn.Dropout(p=0.2),
            nn.Conv2d(n_ch4, n_ch4, kernel_size=(1, 3), stride=1, padding="valid"),
            nn.ReLU(),
            nn.BatchNorm2d(n_ch4, momentum=self.batch_norm_alpha, affine=True, eps=1e-5),
            nn.Dropout(p=0.2),
            nn.AvgPool2d(kernel_size=(1, 7), stride=(1, 7)),
        )
        self.convnet.eval()
        out = self.convnet(torch.zeros(1, 1, input_ch, input_time))

        n_out_time = out.cpu().data.numpy().shape[3]
        self.final_conv_length = n_out_time

        self.n_outputs = out.size()[1] * out.size()[2] * out.size()[3]

        """ Classifier """
        self.clf = nn.Sequential(nn.Linear(self.n_outputs, 50),
                                 nn.ReLU(),
                                 nn.Linear(50, self.n_classes),
                                 nn.Sigmoid()
                                 )
    def forward(self, x):
        output = self.convnet(x)
        output = output.view(output.size()[0], -1)
        output = self.clf(output)
        return output

    def get_embedding(self, x):
        return self.forward(x)