use image::io::Reader;
use std::fs;
use std::path::PathBuf;

use crate::fruits_dataset::{HEIGHT, WIDTH};
use std::collections::{HashMap, HashSet};

pub fn load_data(data_path: &str) -> (Vec<PathBuf>, Vec<usize>, Vec<String>) {
    let mut files = Vec::new();
    let mut labels = Vec::new();
    let mut class_names = Vec::new();

    for entry in fs::read_dir(data_path).unwrap() {
        let entry = entry.unwrap();
        let path = entry.path();
        if path.is_dir() {
            let class_name = path.file_name().unwrap().to_string_lossy().to_string();
            let class_idx = class_names.len();
            class_names.push(class_name.clone());

            for entry in fs::read_dir(&path).unwrap() {
                let entry = entry.unwrap();
                let file_path = entry.path();
                if file_path.is_file() {
                    files.push(file_path);
                    labels.push(class_idx);
                }
            }
        }
    }

    (files, labels, class_names)
}

pub fn balance_y_train_and_test(
    x_train: Vec<PathBuf>,
    y_train: Vec<usize>,
    x_test: Vec<PathBuf>,
    y_test: Vec<usize>,
    target_labels_train: Vec<String>,
    target_labels_test: Vec<String>,
) -> (
    Vec<PathBuf>,
    Vec<usize>,
    Vec<PathBuf>,
    Vec<usize>,
    Vec<String>,
) {
    let label_names_train: Vec<&String> =
        y_train.iter().map(|&i| &target_labels_train[i]).collect();
    let label_names_test: Vec<&String> = y_test.iter().map(|&i| &target_labels_test[i]).collect();

    let set_train: HashSet<&String> = label_names_train.iter().cloned().collect();
    let set_test: HashSet<&String> = label_names_test.iter().cloned().collect();
    let common_labels: Vec<&String> = set_train.intersection(&set_test).cloned().collect();

    let common_labels_set: HashSet<&String> = common_labels.iter().cloned().collect();

    let mut x_train_bal = Vec::new();
    let mut y_train_bal_labels = Vec::new();
    for (x, label) in x_train.into_iter().zip(label_names_train.iter()) {
        if common_labels_set.contains(label) {
            x_train_bal.push(x);
            y_train_bal_labels.push((*label).clone());
        }
    }

    let mut x_test_bal = Vec::new();
    let mut y_test_bal_labels = Vec::new();
    for (x, label) in x_test.into_iter().zip(label_names_test.iter()) {
        if common_labels_set.contains(label) {
            x_test_bal.push(x);
            y_test_bal_labels.push((*label).clone());
        }
    }

    let mut sorted_common_labels: Vec<String> = common_labels.into_iter().cloned().collect();
    sorted_common_labels.sort();
    let label_to_new_index: HashMap<&String, usize> = sorted_common_labels
        .iter()
        .enumerate()
        .map(|(idx, label)| (label, idx))
        .collect();

    let y_train_bal: Vec<usize> = y_train_bal_labels
        .iter()
        .map(|label| *label_to_new_index.get(label).unwrap())
        .collect();
    let y_test_bal: Vec<usize> = y_test_bal_labels
        .iter()
        .map(|label| *label_to_new_index.get(label).unwrap())
        .collect();

    (
        x_train_bal,
        y_train_bal,
        x_test_bal,
        y_test_bal,
        sorted_common_labels,
    )
}

pub fn divide_set<T: Clone>(set: &[T], middle: usize) -> (Vec<T>, Vec<T>) {
    let first = set[..middle].to_vec();
    let second = set[middle..].to_vec();
    (first, second)
}

pub fn data_preprocessing(
    x_train: Vec<PathBuf>,
    x_valid: Vec<PathBuf>,
    x_test: Vec<PathBuf>,
) -> (
    Vec<[[[f32; WIDTH]; HEIGHT]; 3]>,
    Vec<[[[f32; WIDTH]; HEIGHT]; 3]>,
    Vec<[[[f32; WIDTH]; HEIGHT]; 3]>,
) {
    fn convert_images_to_vec(files: &[PathBuf]) -> Vec<[[[f32; WIDTH]; HEIGHT]; 3]> {
        let mut images_vec = Vec::with_capacity(files.len());
        for file in files {
            let img = Reader::open(file)
                .expect("Failed to open image")
                .decode()
                .expect("Failed to decode image")
                .to_rgb8();
            let raw = img.into_raw();

            let mut image = [[[0f32; WIDTH]; HEIGHT]; 3];
            for c in 0..3 {
                for h in 0..HEIGHT {
                    for w in 0..WIDTH {
                        let idx = (h * WIDTH + w) * 3 + c;
                        image[c][h][w] = raw[idx] as f32 / 255.0;
                    }
                }
            }
            images_vec.push(image);
        }
        images_vec
    }

    let x_train = convert_images_to_vec(&x_train);
    let x_valid = convert_images_to_vec(&x_valid);
    let x_test = convert_images_to_vec(&x_test);

    println!(
        "Training set shape: ({}, {}, {}, {})",
        x_train.len(),
        HEIGHT,
        WIDTH,
        3
    );
    println!(
        "Validation set shape: ({}, {}, {}, {})",
        x_valid.len(),
        HEIGHT,
        WIDTH,
        3
    );
    println!(
        "Test set shape: ({}, {}, {}, {})",
        x_test.len(),
        HEIGHT,
        WIDTH,
        3
    );

    (x_train, x_valid, x_test)
}
