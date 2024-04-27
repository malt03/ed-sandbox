use super::{
    differentiable_fn::{PassThrough, Sigmoid},
    layer::Layer,
    util::duplicate_elements,
};
use rand::{rngs::StdRng, SeedableRng};

#[derive(Debug)]
pub struct Mnist {
    first_layer: Layer<Sigmoid>,
    layers: Vec<Layer<Sigmoid>>,
    last_layer: Layer<PassThrough>,
}

impl Mnist {
    pub fn new(layer_num: usize, neural_num: usize) -> Self {
        let mut rng = StdRng::seed_from_u64(42);
        Mnist {
            first_layer: Layer::new(&mut rng, 784 * 2, neural_num),
            layers: (0..layer_num)
                .map(|_| Layer::new(&mut rng, neural_num, neural_num))
                .collect(),
            last_layer: Layer::new(&mut rng, neural_num, 1),
        }
    }

    pub fn forward(&mut self, inputs: &[f64]) -> f64 {
        let x = duplicate_elements(inputs.into_iter()).collect();
        let x = self.first_layer.forward(x);
        let x = self.layers.iter_mut().fold(x, |x, layer| layer.forward(x));
        self.last_layer.forward(x)[0]
    }

    pub fn forward_without_train(&self, inputs: &[f64]) -> f64 {
        let x = duplicate_elements(inputs.into_iter()).collect();
        let x = self.first_layer.forward_without_train(x);
        let x = self
            .layers
            .iter()
            .fold(x, |x, layer| layer.forward_without_train(x));
        self.last_layer.forward_without_train(x)[0]
    }

    pub fn backward(&mut self, delta: f64) {
        self.first_layer.backward(delta);
        self.layers
            .iter_mut()
            .for_each(|layer| layer.backward(delta));
        self.last_layer.backward(delta);
    }
}
