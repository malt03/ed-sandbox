use rand::{rngs::StdRng, Rng, SeedableRng};
use std::f64::consts::E;

fn sigmoid(x: f64) -> f64 {
    1.0 / (1.0 + E.powf(-x))
}

fn sigmoid_derivative(x: f64) -> f64 {
    sigmoid(x) * (1.0 - sigmoid(x))
}

#[derive(Debug, PartialEq, Eq)]
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

#[derive(Debug)]
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

#[derive(Debug)]
struct Layer {
    neurons: Vec<Vec<Neuron>>,
    last_sums: Vec<f64>,
}

impl Layer {
    fn new<R>(rng: &mut R, input: usize, output: usize) -> Self
    where
        R: Rng,
    {
        let neurons: Vec<Vec<_>> = (0..output)
            .map(|i| (0..input).map(|j| Neuron::new(rng, i, j)).collect())
            .collect();
        Layer {
            neurons,
            last_sums: Vec::new(),
        }
    }

    fn forward(&mut self, input: Vec<f64>) -> Vec<f64> {
        self.last_sums = self
            .neurons
            .iter()
            .map(|neurons| {
                neurons
                    .iter()
                    .zip(input.iter())
                    .map(|(neuron, input)| neuron.forward(*input))
                    .sum()
            })
            .collect();

        self.last_sums.iter().map(|sum| sigmoid(*sum)).collect()
    }

    fn backward(&mut self, loss: f64) {
        for (neurons, last_sum) in self.neurons.iter_mut().zip(self.last_sums.iter()) {
            let delta = sigmoid_derivative(*last_sum) * loss;

            for (i, neuron) in neurons.iter_mut().enumerate() {
                if loss > 0. {
                    if i % 2 == 0 {
                        neuron.append_weight(delta);
                    }
                } else {
                    if i % 2 == 1 {
                        neuron.append_weight(-delta);
                    }
                }
            }
        }
    }
}

fn main() {
    let input = [[0., 0.], [0., 1.], [1., 0.], [1., 1.]];
    let targets = [0., 1., 1., 0.];

    let mut rng = StdRng::seed_from_u64(0);
    let mut layer0 = Layer::new(&mut rng, 4, 8);
    let mut layer1 = Layer::new(&mut rng, 8, 1);

    let mut count = 0;
    loop {
        count += 1;
        let mut err = 0.;

        for (&input, &target) in input.iter().zip(targets.iter()) {
            let output =
                layer1.forward(layer0.forward(vec![input[0], input[0], input[1], input[1]]));
            let loss = target - output[0];
            layer1.backward(loss);
            layer0.backward(loss);

            println!(
                "{}, {} -> {:.5}, {:.0}",
                input[0], input[1], output[0], target,
            );

            err += loss.abs();
        }

        println!("err: {:.5}", err);
        if err < 0.1 {
            break;
        }
    }

    println!("count: {}", count);
}
