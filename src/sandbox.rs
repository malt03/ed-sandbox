use ed::{
    duplicate_elements, CrossEntropyLoss, DifferentiableFn, MultiOutputLayer, PassThrough, Sigmoid,
};

use rand::{rngs::StdRng, SeedableRng};

pub struct Gate {
    layer0: MultiOutputLayer<Sigmoid>,
    last_layer: MultiOutputLayer<PassThrough>,
}

impl Gate {
    fn new() -> Self {
        let mut rng = StdRng::seed_from_u64(42);
        Gate {
            layer0: MultiOutputLayer::new(&mut rng, 2, 4, 16),
            last_layer: MultiOutputLayer::new(&mut rng, 2, 16, 1),
        }
    }

    fn forward(&mut self, inputs: &Vec<f64>) -> Vec<f64> {
        let x = duplicate_elements(inputs.into_iter()).collect();
        let x = vec![x; 2];
        let x = self.layer0.forward(x);
        let x = self.last_layer.forward(x);
        x.into_iter().map(|x| x[0]).collect()
    }

    fn forward_without_train(&self, inputs: &[f64]) -> Vec<f64> {
        let x = duplicate_elements(inputs.into_iter()).collect();
        let x = vec![x; 2];
        let x = self.layer0.forward_without_train(x);
        let x = self.last_layer.forward_without_train(x);
        x.into_iter().map(|x| x[0]).collect()
    }

    fn backward(&mut self, deltas: Vec<f64>) {
        self.layer0.backward(&deltas);
        self.last_layer.backward(&deltas);
    }
}

const LEARNING_RATE: f64 = 0.2;

fn main() {
    let mut model = Gate::new();

    // xor
    let train = vec![
        (vec![0., 0.], vec![1., 0.]),
        (vec![1., 0.], vec![0., 1.]),
        (vec![0., 1.], vec![0., 1.]),
        (vec![1., 1.], vec![1., 0.]),
    ];

    for _ in 0..1000 {
        let mut sum_loss = 0.;
        for (input, target) in train.iter() {
            let output = model.forward(input);
            let loss = CrossEntropyLoss::eval((&output, target));
            sum_loss += loss;
            let deltas = CrossEntropyLoss::derivative((&output, target));
            model.backward(
                deltas
                    .into_iter()
                    .map(|delta| delta * LEARNING_RATE)
                    .collect(),
            );
        }

        println!("loss: {}", sum_loss / (train.len() * 2) as f64);
    }

    for (input, _) in train.iter() {
        let outputs = model.forward_without_train(input);
        let outputs = CrossEntropyLoss::softmax(&outputs);
        print!("{:?} -> {:?}", input, outputs);
        println!()
    }
}
