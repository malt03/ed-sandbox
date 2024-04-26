use super::{
    differentiable_fn::{DifferentiableFn, Sigmoid},
    layer::Layer,
};
use rand::{rngs::StdRng, SeedableRng};

pub struct Gate<LastActivation>
where
    LastActivation: DifferentiableFn<Args = f64>,
{
    layer0: Layer<Sigmoid>,
    layer1: Layer<Sigmoid>,
    layer2: Layer<LastActivation>,
}

impl<LastActivation> Gate<LastActivation>
where
    LastActivation: DifferentiableFn<Args = f64>,
{
    pub fn new() -> Self {
        let mut rng = StdRng::seed_from_u64(42);
        Gate {
            layer0: Layer::new(&mut rng, 4, 8),
            layer1: Layer::new(&mut rng, 8, 8),
            layer2: Layer::new(&mut rng, 8, 1),
        }
    }

    pub fn forward(&mut self, inputs: &[f64]) -> f64 {
        let x = vec![inputs[0], inputs[0], inputs[1], inputs[1]];
        let x = self.layer0.forward(x);
        let x = self.layer1.forward(x);
        self.layer2.forward(x)[0]
    }

    pub fn backward(&mut self, delta: f64) {
        self.layer0.backward(delta);
        self.layer1.backward(delta);
        self.layer2.backward(delta);
    }
}

#[cfg(test)]
mod tests {
    use super::super::{
        differentiable_fn::{DifferentiableFn, PassThrough},
        loss_fn::{BCELoss, BCEWithLogitsLoss, MSELoss},
    };
    use super::*;

    #[test]
    fn test_bce_loss() {
        let inputs = [[0., 0.], [0., 1.], [1., 0.], [1., 1.]];
        let targets = [0., 1., 1., 0.];

        let mut xor = Gate::<Sigmoid>::new();

        for _ in 0..200 {
            let mut loss = 0.;

            for (inputs, &target) in inputs.iter().zip(targets.iter()) {
                let output = xor.forward(inputs);
                let delta = BCELoss::derivative((output, target));
                xor.backward(delta * 0.5);

                loss += BCELoss::eval((output, target));
            }

            println!("loss: {:.8}", loss / 4.);
        }

        for (inputs, &target) in inputs.iter().zip(targets.iter()) {
            let output = xor.forward(inputs);
            println!(
                "{}, {} -> {:.8}, {:.0}",
                inputs[0], inputs[1], output, target
            );
        }
    }

    #[test]
    fn test_mse_loss() {
        let inputs = [[0., 0.], [0., 1.], [1., 0.], [1., 1.]];
        let targets = [0., 1., 1., 0.];

        let mut xor = Gate::<Sigmoid>::new();

        for _ in 0..200 {
            let mut loss = 0.;

            for (inputs, &target) in inputs.iter().zip(targets.iter()) {
                let output = xor.forward(inputs);
                let delta = MSELoss::derivative((output, target));
                xor.backward(delta * 0.5);

                loss += MSELoss::eval((output, target));
            }

            println!("loss: {:.8}", loss / 4.);
        }

        for (inputs, &target) in inputs.iter().zip(targets.iter()) {
            let output = xor.forward(inputs);
            println!(
                "{}, {} -> {:.8}, {:.0}",
                inputs[0], inputs[1], output, target
            );
        }
    }

    #[test]
    fn test_bce_with_logits_loss() {
        let inputs = [[0., 0.], [0., 1.], [1., 0.], [1., 1.]];
        let targets = [0., 1., 1., 0.];

        let mut xor = Gate::<PassThrough>::new();

        for _ in 0..200 {
            let mut loss = 0.;

            for (inputs, &target) in inputs.iter().zip(targets.iter()) {
                let output = xor.forward(inputs);
                let delta = BCEWithLogitsLoss::derivative((output, target));
                xor.backward(delta * 0.5);

                loss += BCEWithLogitsLoss::eval((output, target));
            }

            println!("loss: {:.8}", loss / 4.);
        }

        for (inputs, &target) in inputs.iter().zip(targets.iter()) {
            let output = xor.forward(inputs);
            println!(
                "{}, {} -> {:.8}, {:.0}",
                inputs[0],
                inputs[1],
                Sigmoid::eval(output),
                target
            );
        }
    }
}
