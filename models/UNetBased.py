import torch
import numpy as np
import torch.nn as nn



#######################################################################################################################
class ExtractorBlock (nn.Module):
    def __init__ (self, in_c, out_c):
        super(ExtractorBlock, self).__init__()

        self.extractor_block = nn.Sequential(
            nn.Conv1d(in_c, out_c, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.MaxPool1d(kernel_size=2, stride=2)
        )

        return

    
    def forward (self, X):
        return self.extractor_block(X)
#######################################################################################################################
    


#######################################################################################################################
class Extractor (nn.Module):
    def __init__ (self, in_c, out_c, num_extractors=3):
        super(Extractor, self).__init__()

        extractors = []
        for i in range(1, num_extractors+1):
            if i == 1:
                extractors.append(ExtractorBlock(in_c, out_c))
            else:
                extractors.append(ExtractorBlock(out_c, out_c))

        self.extractor = nn.Sequential(*extractors)

        return
    

    def forward (self, X):
        return self.extractor(X)
#######################################################################################################################



#######################################################################################################################
class ConvBlock(nn.Module):
    def __init__(self, in_c, out_c, kernel_size=3, use_batchnorm=False, activation='relu'):
        super(ConvBlock, self).__init__()

        layers = []
        layers.append(nn.Conv1d(in_c, out_c, kernel_size, stride=1, padding=1))

        if use_batchnorm:
            layers.append(nn.BatchNorm1d(out_c))

        if activation == 'relu':
            layers.append(nn.ReLU())
        elif activation == 'leakyrelu':
            layers.append(nn.LeakyReLU())
        elif activation == 'tanh':
            layers.append(nn.Tanh())
        else:
            raise ValueError("Invalid activation function. Supported activations: 'relu', 'leakyrelu', 'tanh'.")

        self.conv_block = nn.Sequential(*layers)

        return
    


    def forward(self, X):
        return self.conv_block(X)
#######################################################################################################################
    


#######################################################################################################################
class UNetEncoderBlock (nn.Module):
    def __init__(self, in_c, out_c):
        super(UNetEncoderBlock, self).__init__()

        self.conv_blocks = nn.Sequential(
            ConvBlock(in_c, out_c),
            ConvBlock(out_c, out_c),
        )
        
        self.pooling = nn.MaxPool1d(kernel_size=2, stride=2)

        return


    def forward(self, X):
        res = self.conv_blocks(X)
        out = self.pooling(res)

        return out, res
#######################################################################################################################
    


#######################################################################################################################
class UNetDecoderBlock (nn.Module):
    def __init__(self, in_c, out_c, use_batchnorm=False, activation='relu'):
        super(UNetDecoderBlock, self).__init__()

        self.upsample = nn.ConvTranspose1d(in_c, out_c, kernel_size=2, stride=2)

        self.conv_blocks = nn.Sequential(
            ConvBlock(in_c, out_c),
            ConvBlock(out_c, out_c),
        )

        return


    def forward(self, X, residual):
        out = self.upsample(X)
        out = self.conv_blocks(torch.cat([residual, out], dim=1))

        return out
#######################################################################################################################


















#######################################################################################################################
class UNetBased (nn.Module):
    def __init__(self, in_c=26, out_c=1, hidden_c=64, seq_len=256, pred_len=32):
        super(UNetBased, self).__init__()

        self.extraction = Extractor(in_c, hidden_c, num_extractors=int(np.log2(seq_len/pred_len)))

        self.encoder = nn.ModuleList([
            UNetEncoderBlock(hidden_c, hidden_c),
            UNetEncoderBlock(hidden_c, hidden_c*2),
            UNetEncoderBlock(hidden_c*2, hidden_c*4),
            UNetEncoderBlock(hidden_c*4, hidden_c*8),
        ])

        self.bottleneck = nn.Sequential(
            ConvBlock(hidden_c*8, hidden_c*16),
            ConvBlock(hidden_c*16, hidden_c*16),
        )

        self.decoder = nn.ModuleList([
            UNetDecoderBlock(hidden_c*16, hidden_c*8),
            UNetDecoderBlock(hidden_c*8, hidden_c*4),
            UNetDecoderBlock(hidden_c*4, hidden_c*2),
            UNetDecoderBlock(hidden_c*2, hidden_c),
        ])

        self.final = nn.Conv1d(hidden_c, out_c, kernel_size=1)

        return
    

    def forward (self, X):
        out = self.extraction(X)
        print(out.shape)

        residuals = []
        for encoder_block in self.encoder:
            out, res = encoder_block(out)
            residuals.append(res)
            print(out.shape, res.shape)

        out = self.bottleneck(out)
        print(out.shape)

        for i, decoder_block in enumerate(self.decoder):
            out = decoder_block(out, residuals[-i-1])
            print(out.shape)

        out = self.final(out)

        return out
#######################################################################################################################