pub mod batcher;
pub mod data_processing;
pub mod fruits_dataset;
mod inference;
pub mod model;
mod time_measuring;
mod training;
mod visualisation;

use crate::{
    data_processing::{balance_y_train_and_test, data_preprocessing, divide_set, load_data},
    fruits_dataset::FruitsDataset,
    inference::{accuracy_score, infer},
    model::ModelConfig,
    time_measuring::TimeMeasuring,
    visualisation::visualization_with_prediction,
};
use burn::{backend::Autodiff, optim::RmsPropConfig};
use burn_tch::{LibTorch, LibTorchDevice};

use training::TrainingConfig;

fn main() {
    let train_dir = "../../datasets/dataset-fruits-360/Training";
    let test_dir = "../../datasets/dataset-fruits-360/Test";

    let mut tm = TimeMeasuring::new("./artifacts/cnn_time_log_");
    let (x_train, y_train, target_labels_train) = load_data(train_dir);
    let (x_test, y_test, target_labels_test) = load_data(test_dir);
    tm.took("Loading the dataset");

    let (x_train, y_train, x_test, y_test, target_labels) = balance_y_train_and_test(
        x_train,
        y_train,
        x_test,
        y_test,
        target_labels_train,
        target_labels_test,
    );
    tm.took("Balancing the dataset");

    tm.reset();
    let (x_valid, x_test) = divide_set(&x_test, 7000);
    let (y_valid, y_test) = divide_set(&y_test, 7000);
    tm.took("Splitting test set into validation- and test set");

    let (x_train, x_valid, x_test) = data_preprocessing(x_train, x_valid, x_test);
    tm.took("Data Preprocessing");

    type MyBackend = LibTorch<f32, i8>;
    type MyAutodiffBackend = Autodiff<MyBackend>;

    let device = LibTorchDevice::Cuda(2);
    let artifact_dir = "./artifacts";

    let dataset_train = FruitsDataset::new(x_train.clone(), y_train);
    let dataset_valid = FruitsDataset::new(x_valid, y_valid);
    println!("Dataset is ready for batching.");
    println!("Shape dataset_train: {}", dataset_train.shape());
    println!("Shape dataset_valid: {}", dataset_valid.shape());
    crate::training::train::<MyAutodiffBackend>(
        artifact_dir,
        dataset_train,
        dataset_valid,
        TrainingConfig::new(ModelConfig::new(196), RmsPropConfig::new()),
        device.clone(),
    );
    tm.took("Training CNN and Saving Model");

    let dataset_test = FruitsDataset::new(x_test, y_test);
    let (preds, targets) = infer::<MyBackend>(artifact_dir, device, dataset_test);
    tm.took("Prediction");

    tm.save_log();

    accuracy_score(&preds, &targets);

    visualization_with_prediction(x_train, &targets, &preds, &target_labels);
}
