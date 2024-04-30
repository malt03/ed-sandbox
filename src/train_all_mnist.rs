use ed::{dataset, duplicate_elements, CrossEntropyLoss, MultiOutputLayer, PassThrough, Sigmoid};
use plotters::prelude::*;
use rand::{rngs::StdRng, SeedableRng};

const LEARNING_RATE: f64 = 0.02;

pub struct Mnist {
    layer0: MultiOutputLayer<Sigmoid>,
    last_layer: MultiOutputLayer<PassThrough>,
}

impl Mnist {
    fn new() -> Self {
        let mut rng = StdRng::seed_from_u64(42);
        Mnist {
            layer0: MultiOutputLayer::new(&mut rng, 10, 784 * 2, 4),
            last_layer: MultiOutputLayer::new(&mut rng, 10, 4, 1),
        }
    }

    fn forward(&mut self, inputs: &[f64]) -> Vec<f64> {
        let x = duplicate_elements(inputs.into_iter()).collect();
        let x = vec![x; 10];
        let x = self.layer0.forward(x);
        let x = self.last_layer.forward(x);
        x.into_iter().map(|x| x[0]).collect()
    }

    fn forward_without_train(&self, inputs: &[f64]) -> Vec<f64> {
        let x = duplicate_elements(inputs.into_iter()).collect();
        let x = vec![x; 10];
        let x = self.layer0.forward_without_train(x);
        let x = self.last_layer.forward_without_train(x);
        x.into_iter().map(|x| x[0]).collect()
    }

    fn backward(&mut self, deltas: Vec<f64>) {
        self.layer0.backward(&deltas);
        self.last_layer.backward(&deltas);
    }
}

fn one_hot_encoding(label: u8) -> Vec<f64> {
    let mut v = vec![0.; 10];
    v[label as usize] = 1.;
    v
}

fn one_hot_encoding_all_labels(data: Vec<(u8, Vec<f64>)>) -> Vec<(u8, Vec<f64>, Vec<f64>)> {
    data.into_iter()
        .map(|(label, image)| (label, one_hot_encoding(label), image))
        .collect()
}

fn main() {
    let mut model = Mnist::new();
    let mnist = dataset::read_mnist();

    let train = one_hot_encoding_all_labels(mnist.train);
    let test = mnist.test;

    let train_len = train.len();

    let mut losses = vec![];
    let mut accuracies = vec![];
    let mut test_accuracies = vec![];

    for _ in 0..100 {
        let mut sum_loss = 0.;
        let mut correct_count = 0;

        for (i, (label, encoded_label, image)) in train.iter().enumerate() {
            if i % 10000 == 0 {
                println!("{} / {}", i, train_len);
            }
            let output = model.forward(image);
            let deltas = CrossEntropyLoss::derivative((&output, encoded_label));
            model.backward(
                deltas
                    .into_iter()
                    .map(|delta| delta * LEARNING_RATE)
                    .collect(),
            );

            let loss = CrossEntropyLoss::eval((&output, encoded_label));
            sum_loss += loss;

            let output = CrossEntropyLoss::softmax(&output);
            let output_index = output
                .iter()
                .enumerate()
                .max_by(|(_, a), (_, b)| a.partial_cmp(b).unwrap())
                .unwrap()
                .0;
            if output_index == *label as usize {
                correct_count += 1;
            }
        }

        let test_correct_count = test
            .iter()
            .filter(|(label, image)| {
                let output = model.forward_without_train(image);
                let output = CrossEntropyLoss::softmax(&output);
                let output_index = output
                    .iter()
                    .enumerate()
                    .max_by(|(_, a), (_, b)| a.partial_cmp(b).unwrap())
                    .unwrap()
                    .0;
                output_index == *label as usize
            })
            .count();

        let loss = sum_loss / train_len as f64;
        let accuracy = correct_count as f64 / train_len as f64;
        let test_accuracy = test_correct_count as f64 / test.len() as f64;
        println!(
            "loss: {:.8}, correct: {} / {} = {}, test: {} / {} = {}",
            loss,
            correct_count,
            train_len,
            accuracy,
            test_correct_count,
            test.len(),
            test_accuracy
        );

        losses.push(sum_loss / train_len as f64);
        accuracies.push(accuracy);
        test_accuracies.push(test_accuracy);
    }

    let root = BitMapBackend::new("plot.png", (1080, 720)).into_drawing_area();
    root.fill(&WHITE).unwrap();
    let loss_range = *min(losses.iter())..*max(losses.iter());
    let accuracy_range = 0.8..1.0;
    let mut chart = ChartBuilder::on(&root)
        .margin(10)
        .x_label_area_size(16)
        .y_label_area_size(42)
        .right_y_label_area_size(42)
        .build_cartesian_2d(0..losses.len() - 1, loss_range)
        .unwrap()
        .set_secondary_coord(0..accuracies.len() - 1, accuracy_range);

    chart.configure_mesh().y_desc("Loss").draw().unwrap();
    chart
        .configure_secondary_axes()
        .y_desc("Accuracy")
        .draw()
        .unwrap();

    let data = losses.iter().enumerate().map(|(i, &loss)| (i, loss));
    let series = LineSeries::new(data, &RED);
    chart.draw_series(series).unwrap();

    let data = accuracies
        .iter()
        .enumerate()
        .map(|(i, &accuracy)| (i, accuracy));
    let series = LineSeries::new(data, &BLUE);
    chart.draw_secondary_series(series).unwrap();

    let data = test_accuracies
        .iter()
        .enumerate()
        .map(|(i, &accuracy)| (i, accuracy));
    let series = LineSeries::new(data, &GREEN);
    chart.draw_secondary_series(series).unwrap();
}

fn max<T, I>(iter: I) -> T
where
    T: PartialOrd,
    I: Iterator<Item = T>,
{
    iter.fold(None, |max, x| match max {
        None => Some(x),
        Some(y) => Some(if x > y { x } else { y }),
    })
    .unwrap()
}

fn min<T, I>(iter: I) -> T
where
    T: PartialOrd,
    I: Iterator<Item = T>,
{
    iter.fold(None, |max, x| match max {
        None => Some(x),
        Some(y) => Some(if x < y { x } else { y }),
    })
    .unwrap()
}
