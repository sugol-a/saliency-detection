from torch import nn
import torchvision

class SaliencyModel(nn.Module):
    def __init__(self):
        super(SaliencyModel, self).__init__()

        vgg = torchvision.models.vgg16(weights=torchvision.models.VGG16_Weights.DEFAULT).features

        # Freeze the model
        # for p in vgg.parameters(recurse=True):
        #     p.requires_grad = False

        # We need the indices from the max-pooling layers inside the
        # pre-trained model for transpose convolution
        vgg[4].return_indices  = True
        vgg[9].return_indices  = True
        vgg[16].return_indices = True
        vgg[23].return_indices = True
        vgg[30].return_indices = True

        self.vgg_conv1 = vgg[0:5]
        self.vgg_conv2 = vgg[5:10]
        self.vgg_conv3 = vgg[10:17]
        self.vgg_conv4 = vgg[17:24]
        self.vgg_conv5 = vgg[24:31]

        self.conv6 = nn.Sequential(
            nn.Conv2d(512, 512, 7, stride=1, padding=3),
            nn.Hardswish(),
            nn.Conv2d(512, 512, 7, stride=1, padding=3),
            nn.Hardswish(),
            nn.Dropout()
        )

        self.unpool_deconv5 = nn.MaxUnpool2d(2, 2)

        self.deconv5 = nn.Sequential(
            # Transpose convolution
            nn.ConvTranspose2d(512, 512, 3, stride=1, padding=1),
            nn.Hardswish(),
        )

        self.unpool_deconv4 = nn.MaxUnpool2d(2, 2)

        self.deconv4 = nn.Sequential(
            nn.ConvTranspose2d(512, 256, 3, stride=1, padding=1),
            nn.Hardswish(),
        )

        self.unpool_deconv3 = nn.MaxUnpool2d(2, 2)

        self.deconv3 = nn.Sequential(
            nn.ConvTranspose2d(256, 128, 3, stride=1, padding=1),
            nn.Hardswish(),
        )

        self.unpool_deconv2 = nn.MaxUnpool2d(2, 2)

        self.deconv2 = nn.Sequential(
            nn.ConvTranspose2d(128, 64, 3, stride=1, padding=1),
            nn.Hardswish()
        )

        self.unpool_deconv1 = nn.MaxUnpool2d(2, 2)

        self.deconv1 = nn.Sequential(
            nn.ConvTranspose2d(64, 32, 3, stride=1, padding=1),
            nn.Hardswish()
        )

        self.score = nn.Sequential(
            nn.Dropout(),
            nn.ConvTranspose2d(32, 1, 3, stride=1, padding=1),
        )

    def forward(self, x):
        x, indices1 = self.vgg_conv1(x)
        x, indices2 = self.vgg_conv2(x)
        x, indices3 = self.vgg_conv3(x)
        x, indices4 = self.vgg_conv4(x)
        x, indices5 = self.vgg_conv5(x)
        x           = self.conv6(x)
        x           = self.unpool_deconv5(x, indices5)
        x           = self.deconv5(x)
        x           = self.unpool_deconv4(x, indices4)
        x           = self.deconv4(x)
        x           = self.unpool_deconv3(x, indices3)
        x           = self.deconv3(x)
        x           = self.unpool_deconv2(x, indices2)
        x           = self.deconv2(x)
        x           = self.unpool_deconv1(x, indices1)
        x           = self.deconv1(x)
        x           = self.score(x)

        return x
