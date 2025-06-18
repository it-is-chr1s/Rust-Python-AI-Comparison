use crate::fruits_dataset::FruitItem;
use burn::{data::dataloader::batcher::Batcher, prelude::*};

#[derive(Clone, Default)]
pub struct FruitsBatcher {}

#[derive(Clone, Debug)]
pub struct FruitsBatch<B: Backend> {
    pub images: Tensor<B, 4>,
    pub targets: Tensor<B, 1, Int>,
}

impl<B: Backend> Batcher<B, FruitItem, FruitsBatch<B>> for FruitsBatcher {
    fn batch(&self, items: Vec<FruitItem>, device: &B::Device) -> FruitsBatch<B> {
        let images: Vec<Tensor<B, 4>> = items
            .iter()
            .map(|item| {
                let data = TensorData::from(item.image).convert::<B::FloatElem>();
                Tensor::<B, 3>::from_data(data, device).unsqueeze::<4>() // [C, H, W] -> [1, C, H, W]
            })
            .collect();

        let labels: Vec<B::IntElem> = items
            .iter()
            .map(|item| (item.label as i64).elem::<B::IntElem>())
            .collect();

        let images = Tensor::cat(images, 0); // [batch_size, C, H, W]

        let targets_data = TensorData::from(&labels[..]);
        let targets = Tensor::<B, 1, Int>::from_data(targets_data, device);

        FruitsBatch { images, targets }
    }
}
