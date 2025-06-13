mod time_measuring;

use linfa::prelude::*;
use linfa_trees::DecisionTree;
use ndarray::{s, Array1, Array2};
use polars::prelude::*;
use std::error::Error;

use crate::time_measuring::TimeMeasuring;

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let mut tm = TimeMeasuring::new("./time_log/cnn_time_log_");

    let melbourne_file_path = "../../datasets/melb_data.csv";
    let df = CsvReader::from_path(melbourne_file_path)?
        .has_header(true)
        .finish()?;
    tm.took("Loading the dataset");

    println!("{}", df.describe(None)?);
    tm.took("Describe for whole dataset");

    let df_clean = df.drop_nulls::<&str>(None)?;
    tm.took("Drop missing values");

    let labels = df_clean
        .column("Price")?
        .f64()?
        .to_owned()
        .into_iter()
        .map(|x| x.unwrap().round() as usize)
        .collect::<Vec<usize>>();
    tm.took("Get single column price");

    let features =
        df_clean.select(&["Rooms", "Bathroom", "Landsize", "Lattitude", "Longtitude"])?;
    tm.took("Selecting features");

    println!("{}", features.describe(None)?);
    tm.took("Describe for modified dataset");

    println!("{:?}", features.head(Some(5)));
    tm.took("Printing head");

    // Convert each feature column to an ndarray-compatible Vec
    let num_cols = features.width();
    let num_rows = features.height();
    let mut combined_features = Vec::with_capacity(num_rows * 5);

    let mut combined_features_inverted = Vec::with_capacity(num_cols);
    for j in 0..num_cols {
        combined_features_inverted.push(extract_feature(&features, j)?);
    }

    for i in 0..num_rows {
        for j in 0..num_cols {
            combined_features.push(combined_features_inverted[j][i]);
        }
    }

    let features_ndarray: Array2<f64> =
        Array2::from_shape_vec((num_rows, num_cols), combined_features)?;
    let labels_ndarray: Array1<usize> = Array1::from_vec(labels);
    tm.took("Converting to Ndarray");

    println!(
        "Features Ndarray:\n{:?}",
        features_ndarray.slice(s![0..5, ..])
    );

    tm.reset();
    let dataset = linfa::dataset::Dataset::new(features_ndarray, labels_ndarray);
    tm.took("Converting to dataset");

    let (train, valid) = dataset.split_with_ratio(0.8);
    tm.took("Splitting in training and validation dataset");

    let model = DecisionTree::params().fit(&train).unwrap();
    tm.took("Training");

    let prediction = model.predict(&valid);
    tm.took("Prediction");

    let actual = valid.targets();

    println!("The 5 validation features and their predictions:");
    for i in 0..5 {
        print!("Features: [");
        for j in 0..valid.records().ncols() {
            print!("{:.2}", valid.records()[(i, j)]);
            if j < valid.records().ncols() - 1 {
                print!(", ");
            }
        }
        println!("]");

        println!("Prediction: {}", prediction[i]);
        println!("Real: {}", actual[i]);
        println!("---");
    }

    tm.reset();
    let abs_diff = (&prediction.mapv(|x| x as i32) - actual.mapv(|x| x as i32)).mapv(|x| x.abs());
    let mae = abs_diff.mean().unwrap();
    tm.took("Calculating MAE");
    println!("Mean Absolute Error: {}", mae);

    tm.save_log();

    Ok(())
}

fn extract_feature(df: &DataFrame, index: usize) -> Result<Vec<f64>, Box<dyn Error>> {
    let column = df
        .select_at_idx(index)
        .ok_or_else(|| format!("Column index {} is out of bounds", index))?;

    let feature = match column.dtype() {
        DataType::Int64 => column
            .i64()?
            .to_owned()
            .into_iter()
            .map(|opt_val| opt_val.unwrap() as f64)
            .collect::<Vec<f64>>(),
        DataType::Float64 => column
            .f64()?
            .to_owned()
            .into_iter()
            .map(|opt_val| opt_val.unwrap())
            .collect::<Vec<f64>>(),
        _ => {
            return Err(Box::from(format!(
                "Unsupported data type for column {}",
                index
            )))
        }
    };

    Ok(feature)
}
