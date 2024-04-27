use ed::{dataset, BCEWithLogitsLoss, DifferentiableFn, Mnist};

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

    for _ in 0..200 {
        let mut loss = 0.;

        for (label, image) in train.iter() {
            let label = float_label(*label);
            let output = model.forward(image);
            let delta = BCEWithLogitsLoss::derivative((output, label));
            model.backward(delta * LEARNING_RATE);

            let l = BCEWithLogitsLoss::eval((output, label)).abs();
            loss += l;
        }

        let correct_count = train
            .iter()
            .filter(|(label, image)| {
                let output = model.forward(image);
                (output > 0.5) == bool_label(*label)
            })
            .count();

        println!(
            "loss: {:.8}, correct: {} / {} = {}",
            loss / train_len as f64,
            correct_count,
            train_len,
            correct_count as f64 / train_len as f64
        );
    }

    run_test(&model, &test);
}
