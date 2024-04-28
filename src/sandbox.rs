use ed::{duplicate_elements, unduplicate_elements, CrossEntropyLoss, Layer, PassThrough, Sigmoid};

use rand::{rngs::StdRng, SeedableRng};

#[derive(Debug)]
pub struct Gate {
    layer0: Layer<Sigmoid>,
    last_layer: Layer<PassThrough>,
}

impl Gate {
    pub fn new() -> Self {
        let mut rng = StdRng::seed_from_u64(42);
        Gate {
            layer0: Layer::new(&mut rng, 4, 16),
            last_layer: Layer::new(&mut rng, 16, 4),
        }
    }

    pub fn forward(&mut self, inputs: &[f64]) -> Vec<f64> {
        let x = duplicate_elements(inputs.into_iter()).collect();
        let x = self.layer0.forward(x);
        let x = self.last_layer.forward(x);
        unduplicate_elements(x.iter()).collect()
    }

    pub fn backward(&mut self, delta: Vec<f64>) {
        let delta: Vec<_> = duplicate_elements(delta.iter()).collect();
        let delta = self.last_layer.backward_multi(&delta);

        for d in delta.iter() {
            self.layer0.backward(*d);
        }
    }
}

const LEARNING_RATE: f64 = 0.8;

fn main() {
    let mut model = Gate::new();

    // xor
    // let train = vec![
    //     (vec![0., 0.], vec![1., 0.]),
    //     (vec![1., 0.], vec![0., 1.]),
    //     (vec![0., 1.], vec![0., 1.]),
    //     (vec![1., 1.], vec![1., 0.]),
    // ];

    // and
    let train = vec![
        (vec![0., 0.], vec![1., 0.]),
        (vec![1., 0.], vec![1., 0.]),
        (vec![0., 1.], vec![1., 0.]),
        (vec![1., 1.], vec![0., 1.]),
    ];

    // or
    // let train = vec![
    //     (vec![0., 0.], vec![1., 0.]),
    //     (vec![1., 0.], vec![0., 1.]),
    //     (vec![0., 1.], vec![0., 1.]),
    //     (vec![1., 1.], vec![0., 1.]),
    // ];

    for _ in 0..1000 {
        let mut sum_loss = 0.;
        for (input, target) in train.iter() {
            let output = model.forward(&input);
            let loss = CrossEntropyLoss::eval((&output, &target));
            let delta = CrossEntropyLoss::derivative((&output, &target));
            model.backward(delta.into_iter().map(|d| d * LEARNING_RATE).collect());

            sum_loss += loss;
        }

        println!("loss: {}", sum_loss / train.len() as f64);
    }

    for (input, _) in train.iter() {
        let output = model.forward(&input);
        print!("{:?} -> ", input);
        for n in CrossEntropyLoss::softmax(&output) {
            print!("{:.2} ", n);
        }
        println!();
    }
}
