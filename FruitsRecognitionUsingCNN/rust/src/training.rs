use crate::{
    batcher::{FruitsBatch, FruitsBatcher},
    fruits_dataset::FruitsDataset,
    model::{Model, ModelConfig},
};

use burn::{
    data::dataloader::DataLoaderBuilder,
    nn::loss::CrossEntropyLossConfig,
    optim::RmsPropConfig,
    prelude::*,
    record::CompactRecorder,
    tensor::backend::AutodiffBackend,
    train::{
        ClassificationOutput, LearnerBuilder, TrainOutput, TrainStep, ValidStep,
        metric::{AccuracyMetric, LossMetric},
    },
};

impl<B: Backend> Model<B> {
    pub fn forward_classification(
        &self,
        images: Tensor<B, 4>,
        targets: Tensor<B, 1, Int>,
    ) -> ClassificationOutput<B> {
        let output = self.forward(images);
        let loss = CrossEntropyLossConfig::new()
            .init(&output.device())
            .forward(output.clone(), targets.clone());

        ClassificationOutput::new(loss, output, targets)
    }
}

impl<B: AutodiffBackend> TrainStep<FruitsBatch<B>, ClassificationOutput<B>> for Model<B> {
    fn step(&self, batch: FruitsBatch<B>) -> TrainOutput<ClassificationOutput<B>> {
        let item = self.forward_classification(batch.images, batch.targets);

        TrainOutput::new(self, item.loss.backward(), item)
    }
}

impl<B: Backend> ValidStep<FruitsBatch<B>, ClassificationOutput<B>> for Model<B> {
    fn step(&self, batch: FruitsBatch<B>) -> ClassificationOutput<B> {
        self.forward_classification(batch.images, batch.targets)
    }
}

#[derive(Config)]
pub struct TrainingConfig {
    pub model: ModelConfig,
    pub optimizer: RmsPropConfig,
    #[config(default = 42)]
    pub seed: u64,
    #[config(default = 64)]
    pub batch_size: usize,
    #[config(default = 1)]
    pub num_workers: usize,
    #[config(default = 30)]
    pub num_epochs: usize,
    #[config(default = 1.0e-3)]
    pub learning_rate: f64,
}

fn create_artifact_dir(artifact_dir: &str) {
    std::fs::remove_dir_all(artifact_dir).ok();
    std::fs::create_dir_all(artifact_dir).ok();
}

pub fn train<B: AutodiffBackend>(
    artifact_dir: &str,
    dataset_train: FruitsDataset,
    dataset_valid: FruitsDataset,
    config: TrainingConfig,
    device: B::Device,
) -> Model<B> {
    println!("Create artifact dir");
    create_artifact_dir(artifact_dir);
    config
        .save(format!("{artifact_dir}/config.json"))
        .expect("Config should be saved successfully");

    println!("Configure seed");
    B::seed(config.seed);

    println!("Initialise batcher");
    let batcher = FruitsBatcher::default();

    println!("Initialise dataloader_train");
    let dataloader_train = DataLoaderBuilder::new(batcher.clone())
        .batch_size(config.batch_size)
        .shuffle(config.seed)
        .num_workers(config.num_workers)
        .build(dataset_train.clone());

    println!("Initialise dataloader_valid");
    let dataloader_valid = DataLoaderBuilder::new(batcher)
        .batch_size(config.batch_size)
        .shuffle(config.seed)
        .num_workers(config.num_workers)
        .build(dataset_valid.clone());

    println!("Initialise learner");
    println!("Batch size: {}", config.batch_size);
    println!("Train dataset size: {}", dataset_train.len());
    println!("Valid dataset size: {}", dataset_valid.len());
    let devices = vec![device.clone()];
    println!("Devices: {:?}", devices);
    println!("Number of devices: {}", devices.len());
    println!("Device type: {}", std::any::type_name::<B::Device>());
    let learner = LearnerBuilder::new(artifact_dir)
        .metric_train_numeric(AccuracyMetric::new())
        .metric_valid_numeric(AccuracyMetric::new())
        .metric_train_numeric(LossMetric::new())
        .metric_valid_numeric(LossMetric::new())
        .devices(vec![device.clone()])
        .num_epochs(config.num_epochs)
        .build(
            config.model.init::<B>(&device),
            config.optimizer.init(),
            config.learning_rate,
        );

    println!("Start fitting");
    let model_trained = learner.fit(dataloader_train, dataloader_valid);

    model_trained
        .clone()
        .save_file(format!("{artifact_dir}/model"), &CompactRecorder::new())
        .expect("Trained model should be saved successfully");

    model_trained
}
