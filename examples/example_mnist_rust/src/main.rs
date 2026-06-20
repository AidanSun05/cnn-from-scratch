use crate::model::*;
use image::{GrayImage, ImageReader};

mod model;

// Specify paths to any number of 28x28 image files
const IMAGE_PATHS: [&str; 4] = [
    "1.png",
    "8.png",
    "6.png",
    "2.png",
];

// Specify full path to the trained model
const MODEL_PATH: &str = "model-mnist.bin";

fn load_image(path: &str) -> Vec<f32> {
    let img = ImageReader::open(path)
        .expect("Could not open image")
        .decode()
        .expect("Could not decode image");

    // Convert to 8-bit grayscale
    let gray: GrayImage = img.to_luma8();

    let width = gray.width();
    let height = gray.height();

    println!("Loaded {}x{} image '{}'.", width, height, path);

    // Convert pixels to f32 normalized [0, 1]
    gray.pixels().map(|p| p[0] as f32 / 255.0).collect()
}

fn main() {
    let _guard = NNMemoryGuard::new();

    let model = Model::load(MODEL_PATH, IMAGE_PATHS.len()).expect("Could not load model");

    let pixels = IMAGE_PATHS.map(load_image).concat();

    let pred = model.predict(&pixels);
    let classes = model.predicted_classes(pred, 10);
    for (i, item) in classes.iter().enumerate() {
        println!(
            "Image {} predicted: {} ({}%)",
            i + 1,
            item.cls,
            item.confidence * 100_f32
        );
    }
}
