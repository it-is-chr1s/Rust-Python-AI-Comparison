use burn::{
    nn::{
        conv::{Conv2d, Conv2dConfig},
        pool::{MaxPool2d, MaxPool2dConfig},
        Dropout, DropoutConfig, Linear, LinearConfig, PaddingConfig2d, Relu,
    },
    prelude::*,
};

/// Expects input of shape [batch_size, 3, 100, 100]
#[derive(Module, Debug)]
pub struct Model<B: Backend> {
    conv1: Conv2d<B>,
    conv2: Conv2d<B>,
    conv3: Conv2d<B>,
    conv4: Conv2d<B>,
    linear1: Linear<B>,
    linear2: Linear<B>,
    dropout1: Dropout,
    dropout2: Dropout,
    activation_relu: Relu,
    pool: MaxPool2d,
}

impl<B: Backend> Model<B> {
    pub fn forward(&self, images: Tensor<B, 4>) -> Tensor<B, 2> {
        let [_batch_size, _rgb_channels, _height, _width] = images.dims();
        let x = images;

        let x = self.conv1.forward(x);
        let x = self.activation_relu.forward(x);
        let x = self.pool.forward(x);

        let x = self.conv2.forward(x);
        let x = self.activation_relu.forward(x);
        let x = self.pool.forward(x);

        let x = self.conv3.forward(x);
        let x = self.activation_relu.forward(x);
        let x = self.pool.forward(x);

        let x = self.conv4.forward(x);
        let x = self.activation_relu.forward(x);
        let x = self.pool.forward(x);

        let x = self.dropout1.forward(x);
        let dims = x.dims();
        let x = x.flatten(1, dims.len() - 1);

        let x = self.linear1.forward(x);
        let x = self.activation_relu.forward(x);

        let x = self.dropout2.forward(x);

        let x = self.linear2.forward(x);

        x
    }
}

#[derive(Config, Debug)]
pub struct ModelConfig {
    num_classes: usize,
}

impl ModelConfig {
    pub fn init<B: Backend>(&self, device: &B::Device) -> Model<B> {
        Model {
            conv1: Conv2dConfig::new([3, 16], [2, 2])
                .with_padding(PaddingConfig2d::Valid)
                .init(device),
            conv2: Conv2dConfig::new([16, 32], [2, 2])
                .with_padding(PaddingConfig2d::Valid)
                .init(device),
            conv3: Conv2dConfig::new([32, 64], [2, 2])
                .with_padding(PaddingConfig2d::Valid)
                .init(device),
            conv4: Conv2dConfig::new([64, 128], [2, 2])
                .with_padding(PaddingConfig2d::Valid)
                .init(device),
            linear1: LinearConfig::new(128 * 5 * 5, 150).init(device),
            linear2: LinearConfig::new(150, self.num_classes).init(device),
            dropout1: DropoutConfig::new(0.3).init(),
            dropout2: DropoutConfig::new(0.4).init(),
            activation_relu: Relu::new(),
            pool: MaxPool2dConfig::new([2, 2]).with_strides([2, 2]).init(),
        }
    }
}
