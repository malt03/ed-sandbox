use super::differentiable_fn::DifferentiableFn;
use rand::Rng;
use std::marker::PhantomData;

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
            NeuronType::Excitatory => self.weight -= delta,
            NeuronType::Inhibitory => self.weight += delta,
        }
    }
}

struct SingleOutputLayer<ActivationFunc>
where
    ActivationFunc: DifferentiableFn<Args = f64>,
{
    neurons: Vec<Neuron>,
    last_output: f64,
    _f: PhantomData<ActivationFunc>,
}

impl<ActivationFunc> SingleOutputLayer<ActivationFunc>
where
    ActivationFunc: DifferentiableFn<Args = f64>,
{
    fn new<R>(rng: &mut R, index: usize, input: usize) -> Self
    where
        R: Rng,
    {
        let neurons: Vec<_> = (0..input).map(|j| Neuron::new(rng, index, j)).collect();
        SingleOutputLayer {
            neurons,
            last_output: 0.,
            _f: PhantomData,
        }
    }

    fn forward(&mut self, inputs: &Vec<f64>) -> f64 {
        self.last_output = self
            .neurons
            .iter()
            .zip(inputs.iter())
            .map(|(neuron, input)| neuron.forward(*input))
            .sum();

        ActivationFunc::eval(self.last_output)
    }

    fn backward(&mut self, delta: f64, last_inputs: &Vec<f64>) {
        let delta = ActivationFunc::derivative(self.last_output) * delta;

        for (i, neuron) in self.neurons.iter_mut().enumerate() {
            if delta < 0. {
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

pub struct Layer<ActivationFunc>
where
    ActivationFunc: DifferentiableFn<Args = f64>,
{
    inner_layers: Vec<SingleOutputLayer<ActivationFunc>>,
    last_inputs: Vec<f64>,
}

impl<ActivationFunc> Layer<ActivationFunc>
where
    ActivationFunc: DifferentiableFn<Args = f64>,
{
    pub fn new<R>(rng: &mut R, input: usize, output: usize) -> Self
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

    pub fn forward(&mut self, inputs: Vec<f64>) -> Vec<f64> {
        let output = self
            .inner_layers
            .iter_mut()
            .map(|layer| layer.forward(&inputs))
            .collect();
        self.last_inputs = inputs;

        output
    }

    pub fn backward(&mut self, delta: f64) {
        for layer in self.inner_layers.iter_mut() {
            layer.backward(delta, &self.last_inputs);
        }
    }
}
