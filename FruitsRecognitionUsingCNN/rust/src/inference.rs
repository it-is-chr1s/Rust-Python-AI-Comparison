use burn::{
    module::Module,
    prelude::*,
    record::{CompactRecorder, Recorder},
    tensor::backend::Backend,
};

use crate::{fruits_dataset::FruitsDataset, training::TrainingConfig};

use crate::batcher::FruitsBatcher;

pub fn infer<B: Backend>(
    artifact_dir: &str,
    device: B::Device,
    dataset_test: FruitsDataset,
) -> (Vec<i32>, Vec<i32>) {
    let config = TrainingConfig::load(format!("{artifact_dir}/config.json"))
        .expect("Config should exist for the model; run train first");
    let record = CompactRecorder::new()
        .load(format!("{artifact_dir}/model").into(), &device)
        .expect("Trained model should exist");

    let model = config.model.init::<B>(&device).load_record(record);

    let dataloader = burn::data::dataloader::DataLoaderBuilder::new(FruitsBatcher::default())
        .batch_size(8)
        .shuffle(0)
        .num_workers(1)
        .build(dataset_test);

    let mut all_predictions = Vec::new();
    let mut all_targets = Vec::new();

    for batch in dataloader.iter() {
        let inputs = batch.images.to_device(&device);
        let targets = batch.targets;

        let outputs = model.forward(inputs);

        let preds: Vec<i32> = outputs
            .argmax(1)
            .into_data()
            .convert::<i32>()
            .into_vec()
            .unwrap();
        let targs: Vec<i32> = targets.into_data().convert::<i32>().into_vec().unwrap();

        all_predictions.extend(preds);
        all_targets.extend(targs);
    }

    (all_predictions, all_targets)
}

pub fn accuracy_score(preds: &[i32], targets: &[i32]) {
    let correct = preds
        .iter()
        .zip(targets.iter())
        .filter(|(p, t)| p == t)
        .count();
    let accuracy = correct as f32 / preds.len() as f32;

    println!("\nTest accuracy: {:.4}", accuracy);
}
