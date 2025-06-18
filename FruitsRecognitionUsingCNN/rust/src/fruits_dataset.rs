use burn::data::dataset::Dataset;
use std::option::Option;
use std::sync::Arc;

#[derive(Clone)]
pub struct FruitsDataset {
    images: Arc<Vec<[[[f32; WIDTH]; HEIGHT]; 3]>>,
    labels: Arc<Vec<usize>>,
}

pub const WIDTH: usize = 100;
pub const HEIGHT: usize = 100;

#[derive(Clone, Debug)]
pub struct FruitItem {
    pub image: [[[f32; WIDTH]; HEIGHT]; 3],
    pub label: usize,
}

impl FruitsDataset {
    pub fn new(images: Vec<[[[f32; WIDTH]; HEIGHT]; 3]>, labels: Vec<usize>) -> Self {
        Self {
            images: Arc::new(images),
            labels: Arc::new(labels),
        }
    }

    pub fn len(&self) -> usize {
        self.images.len()
    }

    pub fn shape(&self) -> String {
        format!(
            "(Images: ({}, {}, {}, {}), Labels: ({}))",
            self.images.len(),
            self.images[0].len(),
            self.images[0][0].len(),
            self.images[0][0][0].len(),
            self.labels.len()
        )
    }
}

impl Dataset<FruitItem> for FruitsDataset {
    fn get(&self, index: usize) -> Option<FruitItem> {
        if index < self.images.len() && index < self.labels.len() {
            Some(FruitItem {
                image: self.images[index].clone(),
                label: self.labels[index],
            })
        } else {
            None
        }
    }

    fn len(&self) -> usize {
        self.len()
    }
}
