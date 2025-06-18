use chrono::Local;
use image::{Rgb, RgbImage};
use imageproc::drawing::draw_text_mut;
use rand::seq::SliceRandom;
use rusttype::{Font, Scale};

pub fn visualization_with_prediction(
    x: Vec<[[[f32; 100]; 100]; 3]>,
    y: &[i32],
    predictions: &[i32],
    target_labels: &[String],
) {
    let mut rng = rand::thread_rng();
    let sample_indices: Vec<usize> = (0..y.len()).collect();
    let sample_indices = sample_indices
        .choose_multiple(&mut rng, 16)
        .cloned()
        .collect::<Vec<_>>();

    let grid_size = 4;
    let img_size = 100;
    let mut grid = RgbImage::new((img_size * grid_size) as u32, (img_size * grid_size) as u32);

    let font_data = include_bytes!("../assets/DejaVuLGCSans.ttf");
    let font = Font::try_from_bytes(font_data as &[u8]).unwrap();
    let scale = Scale::uniform(14.0);

    for (i, &idx) in sample_indices.iter().enumerate() {
        let row = i / grid_size;
        let col = i % grid_size;
        let x_offset = (col * img_size) as i64;
        let y_offset = (row * img_size) as i64;

        let mut img = RgbImage::new(img_size as u32, img_size as u32);
        for y_ in 0..img_size {
            for x_ in 0..img_size {
                let r = (x[idx][0][y_][x_] * 255.0).clamp(0.0, 255.0) as u8;
                let g = (x[idx][1][y_][x_] * 255.0).clamp(0.0, 255.0) as u8;
                let b = (x[idx][2][y_][x_] * 255.0).clamp(0.0, 255.0) as u8;
                img.put_pixel(x_ as u32, y_ as u32, Rgb([r, g, b]));
            }
        }

        image::imageops::overlay(&mut grid, &img, x_offset, y_offset);

        let pred_label = &target_labels[predictions[idx] as usize];
        let true_label = &target_labels[y[idx] as usize];
        let color = if predictions[idx] == y[idx] {
            Rgb([0, 200, 0])
        } else {
            Rgb([200, 0, 0])
        };
        let label = format!("{} ({})", pred_label, true_label);

        draw_text_mut(
            &mut grid,
            color,
            x_offset as i32 + 2,
            y_offset as i32 + 2,
            scale,
            &font,
            &label,
        );
    }

    let timestamp = Local::now().format("%Y%m%d-%H%M%S");
    let path = format!(
        "./artifacts/visualizations/visualization_with_prediction_{}.png",
        timestamp
    );
    std::fs::create_dir_all("./artifacts/visualizations").unwrap();
    grid.save(path).unwrap();
}
