import torch
import torch.nn as nn

class UNet(nn.Module):
    def __init__(self, in_channels = 1, out_channels = 1, load_from = None):
        '''
        Initialize the UNet architecture.
        Args:
            in_channels (int): Number of input channels.
            out_channels (int): Number of output channels.
        '''
        super(UNet, self).__init__()
        ######################
        ### Encoder layers ###
        ######################
        ## con0_conv1_pool1
        self.encode1 = nn.Sequential(
            nn.Conv2d(in_channels, 48, 3, stride=1, padding=1),
            nn.LeakyReLU(negative_slope=0.1),
            nn.Conv2d(48, 48, 3, stride=1, padding=1), 
            nn.LeakyReLU(negative_slope=0.1),
            nn.MaxPool2d(2))
        ##conv2_pool2
        self.encode2 = nn.Sequential(
            nn.Conv2d(48, 48, 3, stride=1, padding=1),
            nn.LeakyReLU(negative_slope=0.1),
            nn.MaxPool2d(2))
        ##conv3_pool3
        self.encode3 = nn.Sequential(
            nn.Conv2d(48, 48, 3, stride=1, padding=1),
            nn.LeakyReLU(negative_slope=0.1),
            nn.MaxPool2d(2))
        ##conv4_pool4
        self.encode4 = nn.Sequential(
            nn.Conv2d(48, 48, 3, stride=1, padding=1),
            nn.LeakyReLU(negative_slope=0.1),
            nn.MaxPool2d(2))
        ##conv5_pool5
        self.encode5 = nn.Sequential(
            nn.Conv2d(48, 48, 3, stride=1, padding=1),
            nn.LeakyReLU(negative_slope=0.1),
            nn.MaxPool2d(2))
        
        ######################
        ### Decoder layers ###
        ######################
        ##conv6_upsample5
        self.encode6 = nn.Sequential(
            nn.Conv2d(48, 48, 3, stride=1, padding=1),
            nn.ConvTranspose2d(48, 48, 3, stride=2, padding=1, output_padding=1))
        ## decon5a_b_upsample4
        self.decode1 = nn.Sequential(
            nn.Conv2d(96, 96, 3, stride=1, padding=1),
            nn.LeakyReLU(negative_slope=0.1),
            nn.Conv2d(96, 96, 3, stride=1, padding=1),
            nn.LeakyReLU(negative_slope=0.1),
            nn.ConvTranspose2d(96, 96, 3, stride=2, padding=1, output_padding=1))
        ## deconv4a_4b_upsample3
        self.decode2 = nn.Sequential(
            nn.Conv2d(144, 96, 3, stride=1, padding=1),
            nn.LeakyReLU(negative_slope=0.1),
            nn.Conv2d(96, 96, 3, stride=1, padding=1),
            nn.LeakyReLU(negative_slope=0.1),
            nn.ConvTranspose2d(96, 96, 3, stride=2, padding=1, output_padding=1)) 
        ##  deconv3a_3b_upsample2
        self.decode3 = nn.Sequential(
            nn.Conv2d(144, 96, 3, stride=1, padding=1),
            nn.LeakyReLU(negative_slope=0.1),
            nn.Conv2d(96, 96, 3, stride=1, padding=1),
            nn.LeakyReLU(negative_slope=0.1),
            nn.ConvTranspose2d(96, 96, 3, stride=2, padding=1, output_padding=1))
        ##  deconv2a_2b_upsample1
        self.decode4 = nn.Sequential(
            nn.Conv2d(144, 96, 3, stride=1, padding=1),
            nn.LeakyReLU(negative_slope=0.1),
            nn.Conv2d(96, 96, 3, stride=1, padding=1),
            nn.LeakyReLU(negative_slope=0.1),
            nn.ConvTranspose2d(96, 96, 3, stride=2, padding=1, output_padding=1))
        ## deconv1a_1b
        self.decode5 = nn.Sequential (
            nn.Conv2d(96 + in_channels, 64, 3, stride=1, padding=1),
            nn.LeakyReLU(negative_slope = 0.1),
            nn.Conv2d(64, 32, 3, stride=1, padding=1),
            nn.LeakyReLU(negative_slope=0.1))
        
        ####################
        ### Output layer ###
        ####################
        self.output_layer = nn.Conv2d(32, out_channels, 3, stride=1, padding=1)

        ## initialize weight
        if load_from is None:
            self._init_weights()
        else:
            self.load_state_dict(torch.load(load_from))
            print('Weights loaded from %s'%load_from)
        
    def forward(self,x):
        '''
        Forward pass through the UNet.
        Args:
            x (torch.Tensor): Input tensor.
        Returns:
            torch.Tensor: Output tensor.
        '''
        pool1 = self.encode1(x)
        pool2 = self.encode2(pool1)
        pool3 = self.encode3(pool2)
        pool4 = self.encode4(pool3)
        pool5 = self.encode5(pool4)


        upsample5 = self.encode6(pool5)
        concat5 = torch.cat((upsample5, pool4), dim=1)
        upsample4 = self.decode1(concat5)
        concat4 = torch.cat((upsample4, pool3), dim=1)
        upsample3 = self.decode2(concat4)
        concat3 = torch.cat((upsample3, pool2), dim=1)
        upsample2 = self.decode3(concat3)
        concat2 = torch.cat((upsample2, pool1), dim=1)
        upsample1 = self.decode4(concat2)
        concat1 = torch.cat((upsample1, x), dim=1)
        upsample0 = self.decode5(concat1)
        output = self.output_layer(upsample0)
        return output
    
    def _init_weights(self):
        """
        Initializes weights using He et al. (2015).
        """
        for m in self.modules():
            if isinstance(m, nn.ConvTranspose2d) or isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight.data)
                nn.init.constant_(m.bias.data, 0)
                


class UNet_Upsample(nn.Module):
    def __init__(self, in_channels = 1, out_channels = 1, mode='nearest', load_from = None):
        '''
        Initialize the UNet architecture.
        Args:
            in_channels (int): Number of input channels.
            out_channels (int): Number of output channels.
        '''
        super(UNet_Upsample, self).__init__()
        ######################
        ### Encoder layers ###
        ######################
        ## con0_conv1_pool1
        self.encode1 = nn.Sequential(
            nn.Conv2d(in_channels, 48, 3, stride=1, padding=1),
            nn.LeakyReLU(negative_slope=0.1),
            nn.Conv2d(48, 48, 3, stride=1, padding=1), 
            nn.LeakyReLU(negative_slope=0.1),
            nn.MaxPool2d(2))
        ##conv2_pool2
        self.encode2 = nn.Sequential(
            nn.Conv2d(48, 48, 3, stride=1, padding=1),
            nn.LeakyReLU(negative_slope=0.1),
            nn.MaxPool2d(2))
        ##conv3_pool3
        self.encode3 = nn.Sequential(
            nn.Conv2d(48, 48, 3, stride=1, padding=1),
            nn.LeakyReLU(negative_slope=0.1),
            nn.MaxPool2d(2))
        ##conv4_pool4
        self.encode4 = nn.Sequential(
            nn.Conv2d(48, 48, 3, stride=1, padding=1),
            nn.LeakyReLU(negative_slope=0.1),
            nn.MaxPool2d(2))
        ##conv5_pool5
        self.encode5 = nn.Sequential(
            nn.Conv2d(48, 48, 3, stride=1, padding=1),
            nn.LeakyReLU(negative_slope=0.1),
            nn.MaxPool2d(2))
        
        ######################
        ### Decoder layers ###
        ######################
        ##conv6_upsample5
        self.encode6 = nn.Sequential(
            nn.Conv2d(48, 48, 3, stride=1, padding=1),
            nn.Upsample(scale_factor=2, mode=mode))
        ## decon5a_b_upsample4
        self.decode1 = nn.Sequential(
            nn.Conv2d(96, 96, 3, stride=1, padding=1),
            nn.LeakyReLU(negative_slope=0.1),
            nn.Conv2d(96, 96, 3, stride=1, padding=1),
            nn.LeakyReLU(negative_slope=0.1),
            nn.Upsample(scale_factor=2, mode=mode))
        ## deconv4a_4b_upsample3
        self.decode2 = nn.Sequential(
            nn.Conv2d(144, 96, 3, stride=1, padding=1),
            nn.LeakyReLU(negative_slope=0.1),
            nn.Conv2d(96, 96, 3, stride=1, padding=1),
            nn.LeakyReLU(negative_slope=0.1),
            nn.Upsample(scale_factor=2, mode=mode))
        ##  deconv3a_3b_upsample2
        self.decode3 = nn.Sequential(
            nn.Conv2d(144, 96, 3, stride=1, padding=1),
            nn.LeakyReLU(negative_slope=0.1),
            nn.Conv2d(96, 96, 3, stride=1, padding=1),
            nn.LeakyReLU(negative_slope=0.1),
            nn.Upsample(scale_factor=2, mode=mode))
        ##  deconv2a_2b_upsample1
        self.decode4 = nn.Sequential(
            nn.Conv2d(144, 96, 3, stride=1, padding=1),
            nn.LeakyReLU(negative_slope=0.1),
            nn.Conv2d(96, 96, 3, stride=1, padding=1),
            nn.LeakyReLU(negative_slope=0.1),
            nn.Upsample(scale_factor=2, mode=mode))
        ## deconv1a_1b
        self.decode5 = nn.Sequential (
            nn.Conv2d(96 + in_channels, 64, 3, stride=1, padding=1),
            nn.LeakyReLU(negative_slope = 0.1),
            nn.Conv2d(64, 32, 3, stride=1, padding=1),
            nn.LeakyReLU(negative_slope=0.1))
        
        ####################
        ### Output layer ###
        ####################
        self.output_layer = nn.Conv2d(32, out_channels, 3, stride=1, padding=1)

        ## initialize weight
        if load_from is None:
            self._init_weights()
        else:
            self.load_state_dict(torch.load(load_from))
            print('Weights loaded from %s'%load_from)

    def forward(self,x):
        '''
        Forward pass through the UNet.
        Args:
            x (torch.Tensor): Input tensor.
        Returns:
            torch.Tensor: Output tensor.
        '''
        pool1 = self.encode1(x)
        pool2 = self.encode2(pool1)
        pool3 = self.encode3(pool2)
        pool4 = self.encode4(pool3)
        pool5 = self.encode5(pool4)


        upsample5 = self.encode6(pool5)
        concat5 = torch.cat((upsample5, pool4), dim=1)
        upsample4 = self.decode1(concat5)
        concat4 = torch.cat((upsample4, pool3), dim=1)
        upsample3 = self.decode2(concat4)
        concat3 = torch.cat((upsample3, pool2), dim=1)
        upsample2 = self.decode3(concat3)
        concat2 = torch.cat((upsample2, pool1), dim=1)
        upsample1 = self.decode4(concat2)
        concat1 = torch.cat((upsample1, x), dim=1)
        upsample0 = self.decode5(concat1)
        output = self.output_layer(upsample0)
        return output
    
    def _init_weights(self):
        """
        Initializes weights using He et al. (2015).
        """
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight.data)
                nn.init.constant_(m.bias.data, 0)