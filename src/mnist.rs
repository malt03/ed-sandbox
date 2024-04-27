use std::fs::File;
use std::io::{self, Read};
use std::path::Path;

fn read_labels<P: AsRef<Path>>(path: P) -> io::Result<Vec<u8>> {
    let mut file = File::open(path)?;
    let mut buffer = [0u8; 8];
    file.read_exact(&mut buffer)?;
    let magic = u32::from_be_bytes(buffer[0..4].try_into().unwrap());
    let size = u32::from_be_bytes(buffer[4..8].try_into().unwrap());

    if magic != 2049 {
        panic!("Magic number mismatch, expected 2049, got {}", magic);
    }

    let mut labels = vec![0u8; size as usize];
    file.read_exact(&mut labels)?;

    Ok(labels)
}

fn read_images_labels<P: AsRef<Path>>(path: P) -> io::Result<Vec<Vec<f64>>> {
    let mut images = Vec::new();

    let mut file = File::open(path)?;
    let mut buffer = [0u8; 16];
    file.read_exact(&mut buffer)?;
    let magic = u32::from_be_bytes(buffer[0..4].try_into().unwrap());
    let size = u32::from_be_bytes(buffer[4..8].try_into().unwrap());
    let rows = u32::from_be_bytes(buffer[8..12].try_into().unwrap());
    let cols = u32::from_be_bytes(buffer[12..16].try_into().unwrap());

    if magic != 2051 {
        panic!("Magic number mismatch, expected 2051, got {}", magic);
    }

    let image_size = (rows * cols) as usize;
    let total_image_data_size = image_size * size as usize;
    let mut image_data = vec![0u8; total_image_data_size];
    file.read_exact(&mut image_data)?;

    for i in 0..size {
        let start = i as usize * image_size;
        let image_slice = &image_data[start..start + image_size];
        images.push(image_slice.to_vec());
    }

    let images = images
        .into_iter()
        .map(|image| {
            image
                .into_iter()
                .map(|pixel| pixel as f64 / 255.0)
                .collect()
        })
        .collect();

    Ok(images)
}

fn read_dataset<P: AsRef<Path>>(labels_path: P, images_path: P) -> io::Result<Vec<(u8, Vec<f64>)>> {
    let labels = read_labels(labels_path)?;
    let images = read_images_labels(images_path)?;

    Ok(labels.into_iter().zip(images.into_iter()).collect())
}

pub struct Mnist {
    pub train: Vec<(u8, Vec<f64>)>,
    pub test: Vec<(u8, Vec<f64>)>,
}

pub fn read_mnist() -> Mnist {
    let train = read_dataset(
        "mnist/train-labels.idx1-ubyte",
        "mnist/train-images.idx3-ubyte",
    )
    .expect("Failed to read training dataset");
    let test = read_dataset(
        "mnist/t10k-labels.idx1-ubyte",
        "mnist/t10k-images.idx3-ubyte",
    )
    .expect("Failed to read test dataset");

    Mnist { train, test }
}
