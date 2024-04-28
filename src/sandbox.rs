use ed::{
    duplicate_elements, unduplicate_elements, BCEWithLogitsLoss, CrossEntropyLoss,
    DifferentiableFn, Layer, PassThrough, Sigmoid,
};

use rand::{rngs::StdRng, SeedableRng};

#[derive(Debug)]
pub struct Sandbox {
    // layer0: Layer<Sigmoid>,
    layer1: Layer<PassThrough>,
}

impl Sandbox {
    pub fn new() -> Self {
        let mut rng = StdRng::seed_from_u64(42);
        Sandbox {
            // layer0: Layer::new(&mut rng, 16, 16),
            layer1: Layer::new(&mut rng, 8, 8),
        }
    }

    pub fn forward(&mut self, inputs: &[f64]) -> Vec<f64> {
        let x = duplicate_elements(inputs.into_iter()).collect();
        // let x = self.layer0.forward(x);
        let x = self.layer1.forward(x);
        unduplicate_elements(x.iter()).collect()
    }

    pub fn forward_without_train(&self, inputs: &[f64]) -> Vec<f64> {
        let x = duplicate_elements(inputs.into_iter()).collect();
        // let x = self.layer0.forward_without_train(x);
        let x = self.layer1.forward_without_train(x);
        unduplicate_elements(x.iter()).collect()
    }

    pub fn backward(&mut self, delta: Vec<f64>) {
        // for d in delta.iter() {
        //     self.layer0.backward(*d);
        // }
        let delta = duplicate_elements(delta.iter()).collect();
        self.layer1.backward_multi(&delta);
        // self.layer1.backward(delta[0]);
    }
}

fn main() {
    let mut model = Sandbox::new();

    let input = vec![0.0, 1.0, 1.0, 1.0];
    let target = vec![0.0, 0.0, 1.0, 0.0];

    for _ in 0..10 {
        let output = model.forward(&input);
        let loss = CrossEntropyLoss::eval((&output, &target));
        let delta = CrossEntropyLoss::derivative((&output, &target));
        println!("{}: {:?}", loss, CrossEntropyLoss::softmax(&output),);
        model.backward(delta);
    }
}
