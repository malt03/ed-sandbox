use rand::{rngs::StdRng, Rng, SeedableRng};
use std::f64::consts::E;

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
    last_sum: f64,
}

impl SingleOutputLayer {
    fn new<R>(rng: &mut R, index: usize, input: usize) -> Self
    where
        R: Rng,
    {
        let neurons: Vec<_> = (0..input).map(|j| Neuron::new(rng, index, j)).collect();
        SingleOutputLayer {
            neurons,
            last_sum: 0.,
        }
    }

    fn forward(&mut self, inputs: &Vec<f64>) -> f64 {
        self.last_sum = self
            .neurons
            .iter()
            .zip(inputs.iter())
            .map(|(neuron, input)| neuron.forward(*input))
            .sum();

        sigmoid(self.last_sum)
    }

    fn backward(&mut self, loss: f64, last_inputs: &Vec<f64>) {
        let delta = sigmoid_derivative(self.last_sum) * loss;

        for (i, neuron) in self.neurons.iter_mut().enumerate() {
            let delta = delta * last_inputs[i];
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

fn main() {
    let inputs = [[0., 0.], [0., 1.], [1., 0.], [1., 1.]];
    let targets = [0., 1., 1., 0.];

    let mut rng = StdRng::seed_from_u64(0);
    let mut layer0 = Layer::new(&mut rng, 4, 8);
    let mut layer1 = Layer::new(&mut rng, 8, 8);
    let mut layer2 = Layer::new(&mut rng, 8, 1);

    let mut count = 0;
    loop {
        count += 1;
        let mut err = 0.;

        for (&inputs, &target) in inputs.iter().zip(targets.iter()) {
            let output = layer2.forward(
                layer1.forward(layer0.forward(vec![inputs[0], inputs[0], inputs[1], inputs[1]])),
            );
            let loss = target - output[0];
            layer2.backward(loss);
            layer1.backward(loss);
            layer0.backward(loss);

            println!(
                "{}, {} -> {:.5}, {:.0}",
                inputs[0], inputs[1], output[0], target,
            );

            err += loss.abs();
        }

        println!("err: {:.5}", err);
        if err < 1e-3 {
            break;
        }
    }

    println!("count: {}", count);
}
