use ed::{dataset, BCEWithLogitsLoss, DifferentiableFn, Mnist};
use plotters::prelude::*;

const LEARNING_RATE: f64 = 0.02;
const FIRST: u8 = 0;
const SECOND: u8 = 1;

fn filter_two_value(dataset: Vec<(u8, Vec<f64>)>) -> Vec<(u8, Vec<f64>)> {
    dataset
        .into_iter()
        .filter(|(label, _)| *label == FIRST || *label == SECOND)
        .collect()
}

fn float_label(label: u8) -> f64 {
    if label == SECOND {
        1.
    } else {
        0.
    }
}

fn bool_label(label: u8) -> bool {
    label == SECOND
}

fn run_test(model: &Mnist, test: &Vec<(u8, Vec<f64>)>) {
    let test_len = test.len();
    let correct_count = test
        .iter()
        .filter(|(label, image)| {
            let output = model.forward_without_train(image);
            (output > 0.5) == bool_label(*label)
        })
        .count();

    println!(
        "correct: {} / {} = {}",
        correct_count,
        test_len,
        correct_count as f64 / test_len as f64
    );
}

fn main() {
    let mut model = Mnist::new(4, 16);
    let mnist = dataset::read_mnist();

    let train: Vec<_> = filter_two_value(mnist.train);
    let test: Vec<_> = filter_two_value(mnist.test);

    let train_len = train.len();

    let first_count = train.iter().filter(|(label, _)| *label == 0).count();
    println!(
        "{}: {}, {}: {}",
        FIRST,
        first_count,
        SECOND,
        train_len - first_count
    );

    run_test(&model, &test);

    let mut losses = vec![];
    let mut accuracies = vec![];

    for _ in 0..200 {
        let mut sum_loss = 0.;

        for (label, image) in train.iter() {
            let label = float_label(*label);
            let output = model.forward(image);
            let delta = BCEWithLogitsLoss::derivative((output, label));
            model.backward(delta * LEARNING_RATE);

            let l = BCEWithLogitsLoss::eval((output, label)).abs();
            sum_loss += l;
        }

        let correct_count = train
            .iter()
            .filter(|(label, image)| {
                let output = model.forward(image);
                (output > 0.5) == bool_label(*label)
            })
            .count();

        let loss = sum_loss / train_len as f64;
        let accuracy = correct_count as f64 / train_len as f64;
        println!(
            "loss: {:.8}, correct: {} / {} = {}",
            loss, correct_count, train_len, accuracy
        );

        losses.push(sum_loss / train_len as f64);
        accuracies.push(accuracy);
    }

    run_test(&model, &test);

    let root = BitMapBackend::new("plot.png", (1080, 720)).into_drawing_area();
    root.fill(&WHITE).unwrap();
    let loss_range = *min(losses.iter())..*max(losses.iter());
    let accuracy_range = *min(accuracies.iter())..*max(accuracies.iter());
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
