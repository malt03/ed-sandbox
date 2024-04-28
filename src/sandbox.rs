use ed::{duplicate_elements, unduplicate_elements, CrossEntropyLoss, Layer, PassThrough, Sigmoid};

use rand::{rngs::StdRng, SeedableRng};

#[derive(Debug)]
pub struct Sandbox {
    layer0: Layer<Sigmoid>,
    last_layer: Layer<PassThrough>,
}

impl Sandbox {
    pub fn new() -> Self {
        let mut rng = StdRng::seed_from_u64(42);
        Sandbox {
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

    // pub fn forward_without_train(&self, inputs: &[f64]) -> Vec<f64> {
    //     let x = duplicate_elements(inputs.into_iter()).collect();
    //     let x = self.layer0.forward_without_train(x);
    //     let x = self.last_layer.forward_without_train(x);
    //     unduplicate_elements(x.iter()).collect()
    // }

    pub fn backward(&mut self, delta: Vec<f64>) {
        let delta = duplicate_elements(delta.iter()).collect();
        let delta = self.last_layer.backward_multi(&delta);

        // for d in unduplicate_elements(delta.iter()) {
        //     self.layer0.backward(d);
        // }
    }
}

fn main() {
    let mut model = Sandbox::new();

    let train = vec![
        // (vec![0., 0.], vec![1., 0.]),
        (vec![1., 0.], vec![0., 1.]),
        (vec![0., 1.], vec![0., 1.]),
        (vec![1., 1.], vec![1., 0.]),
    ];

    for _ in 0..10000 {
        let mut sum_loss = 0.;
        for (input, target) in train.iter() {
            let output = model.forward(&input);
            let loss = CrossEntropyLoss::eval((&output, &target));
            let delta = CrossEntropyLoss::derivative((&output, &target));
            model.backward(delta.into_iter().map(|d| d * 0.2).collect());

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
