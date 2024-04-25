use rand::{rngs::StdRng, Rng, SeedableRng};
use std::{f64::consts::E, vec};

fn sigmoid(x: f64) -> f64 {
    1.0 / (1.0 + E.powf(-x))
}

fn sigmoid_derivative(x: f64) -> f64 {
    let s = sigmoid(x);
    s * (1.0 - s)
}

enum NeuronType {
    Excitatory,
    Inhibitory,
}

impl NeuronType {
    fn new(i: usize, j: usize) -> Self {
        if i % 2 == j % 2 {
            NeuronType::Excitatory
        } else {
            NeuronType::Inhibitory
        }
    }

    fn weight<R>(&self, rng: &mut R) -> f64
    where
        R: Rng,
    {
        match self {
            NeuronType::Excitatory => rng.gen(),
            NeuronType::Inhibitory => -rng.gen::<f64>(),
        }
    }
}

struct Neuron {
    neuron_type: NeuronType,
    weight: f64,
}

impl Neuron {
    fn new<R>(rng: &mut R, i: usize, j: usize) -> Self
    where
        R: Rng,
    {
        let neuron_type = NeuronType::new(i, j);
        Neuron {
            weight: neuron_type.weight(rng),
            neuron_type,
        }
    }

    fn forward(&self, input: f64) -> f64 {
        input * self.weight
    }

    fn append_weight(&mut self, delta: f64) {
        match self.neuron_type {
            NeuronType::Excitatory => self.weight += delta,
            NeuronType::Inhibitory => self.weight -= delta,
        }
    }
}

struct SingleOutputLayer {
    neurons: Vec<Neuron>,
    delta: f64,
}

impl SingleOutputLayer {
    fn new<R>(rng: &mut R, index: usize, input: usize) -> Self
    where
        R: Rng,
    {
        let neurons: Vec<_> = (0..input).map(|j| Neuron::new(rng, index, j)).collect();
        SingleOutputLayer { neurons, delta: 0. }
    }

    fn forward(&mut self, inputs: &Vec<f64>) -> f64 {
        let sum = self
            .neurons
            .iter()
            .zip(inputs.iter())
            .map(|(neuron, input)| neuron.forward(*input))
            .sum();

        self.delta = sigmoid_derivative(sum);
        sigmoid(sum)
    }

    fn backward(&mut self, loss: f64, last_inputs: &Vec<f64>) {
        let delta = self.delta * loss;

        for (i, neuron) in self.neurons.iter_mut().enumerate() {
            if loss > 0. {
                if i % 2 == 0 {
                    neuron.append_weight(delta * last_inputs[i]);
                }
            } else {
                if i % 2 == 1 {
                    neuron.append_weight(-delta * last_inputs[i]);
                }
            }
        }
    }
}

struct Layer {
    inner_layers: Vec<SingleOutputLayer>,
    last_inputs: Vec<f64>,
}

impl Layer {
    fn new<R>(rng: &mut R, input: usize, output: usize) -> Self
    where
        R: Rng,
    {
        Layer {
            inner_layers: (0..output)
                .map(|i| SingleOutputLayer::new(rng, i, input))
                .collect(),
            last_inputs: Vec::new(),
        }
    }

    fn forward(&mut self, inputs: Vec<f64>) -> Vec<f64> {
        let output = self
            .inner_layers
            .iter_mut()
            .map(|layer| layer.forward(&inputs))
            .collect();
        self.last_inputs = inputs;

        output
    }

    fn backward(&mut self, loss: f64) {
        for layer in self.inner_layers.iter_mut() {
            layer.backward(loss, &self.last_inputs);
        }
    }
}

struct XOR {
    layer0: Layer,
    layer1: Layer,
    layer2: Layer,
}

impl XOR {
    fn new() -> Self {
        let mut rng = StdRng::seed_from_u64(0);
        XOR {
            layer0: Layer::new(&mut rng, 4, 8),
            layer1: Layer::new(&mut rng, 8, 8),
            layer2: Layer::new(&mut rng, 8, 1),
        }
    }

    fn forward(&mut self, inputs: &[f64]) -> f64 {
        let x = vec![inputs[0], inputs[0], inputs[1], inputs[1]];
        let x = self.layer0.forward(x);
        let x = self.layer1.forward(x);
        self.layer2.forward(x)[0]
    }

    fn backward(&mut self, loss: f64) {
        self.layer0.backward(loss);
        self.layer1.backward(loss);
        self.layer2.backward(loss);
    }
}

fn main() {
    let inputs = [[0., 0.], [0., 1.], [1., 0.], [1., 1.]];
    let targets = [0., 1., 1., 0.];

    let mut xor = XOR::new();

    for _ in 0..10000 {
        let mut err = 0.;

        for (inputs, &target) in inputs.iter().zip(targets.iter()) {
            let output = xor.forward(inputs);
            let loss = (target - output) * 2.;
            xor.backward(loss);

            println!(
                "{}, {} -> {:.8}, {:.0}",
                inputs[0], inputs[1], output, target,
            );

            err += loss.abs();
        }

        println!("err: {:.8}", err);
    }
}
