import torch
import torch.nn as nn



#######################################################################################################################
class ConvBlock(nn.Module):
    def __init__(self, in_c, out_c, kernel_size=3, use_batchnorm=True, activation='relu'):
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
# Base encoder block
#######################################################################################################################
class ConvEncoderBlock (nn.Module):
    def __init__ (self, in_c, out_c, num_blocks=2):
        super(ConvEncoderBlock, self).__init__()

        blocks = []
        for i in range(1, num_blocks+1):
            if i == 1:
                blocks.append(ConvBlock(in_c, out_c, kernel_size=3, use_batchnorm=False, activation='relu'))
            else:
                blocks.append(ConvBlock(out_c, out_c, kernel_size=3, use_batchnorm=False, activation='relu'))

        # pooling
        blocks.append(nn.MaxPool1d(kernel_size=2, stride=2))

        self.encoder_block = nn.Sequential(*blocks)
        
        return
    
    

    def forward (self, X):        
        return self.encoder_block(X)
#######################################################################################################################
    


#######################################################################################################################
class ConvEncoder (nn.Module):
    def __init__ (self, input_dim=26, hidden_c=64):
        super(ConvEncoder, self).__init__()

        # Возьмём енкодер от модели VGG16
        self.encoder = nn.Sequential(
            ConvEncoderBlock(input_dim, hidden_c, num_blocks=2),
            ConvEncoderBlock(hidden_c, hidden_c*2, num_blocks=2),
            ConvEncoderBlock(hidden_c*2, hidden_c*4, num_blocks=3),
            ConvEncoderBlock(hidden_c*4, hidden_c*8, num_blocks=3),
            ConvEncoderBlock(hidden_c*8, hidden_c*16, num_blocks=3),
            nn.Flatten()
        )
        
        return
        

    def forward (self, X):
        return self.encoder(X)
#######################################################################################################################
    











#######################################################################################################################
class VGG16Based (nn.Module):
    def __init__ (self, input_dim=26, pred_len=30, seq_len=180, hidden_c=64, num_blocks=5):
        super(VGG16Based, self).__init__()

        latent_size = int(seq_len / (2**num_blocks))
        latent_channels = int(hidden_c * (2**(num_blocks-1)))
        
        self.encoder = ConvEncoder(input_dim, hidden_c)

        self.decoder = nn.Sequential(
            nn.Linear(latent_channels * latent_size, 1024),
            nn.ReLU(),
            nn.Linear(1024, 256),
            nn.ReLU(),
            nn.Linear(256, pred_len)
        )


    def forward (self, X):
        Z = self.encode(X) # latent
        out = self.decode(Z) # output
        out = out.view(X.size(0), 1, -1)

        return out
    

    def encode (self, X):
        return self.encoder(X)
    

    def decode (self, Z):
        return self.decoder(Z)
#######################################################################################################################