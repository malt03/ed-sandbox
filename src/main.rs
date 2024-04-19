use rand::{rngs::ThreadRng, Rng};
use rand_distr::Normal;
use std::f64::consts::E;

fn sigmoid(x: f64) -> f64 {
    1.0 / (1.0 + E.powf(-x))
}

fn sigmoid_derivative(x: f64) -> f64 {
    sigmoid(x) * (1.0 - sigmoid(x))
}

#[derive(Debug)]
struct NeuralNetwork {
    w: [[f64; 2]; 3],
    b: [f64; 3],
}

#[derive(Debug, Clone, Copy)]
struct ForwardResult {
    h: [f64; 2],
    y: f64,
}

fn kaiming_init(rng: &mut ThreadRng, rows: usize) -> f64 {
    let std_dev = (2.0 / rows as f64).sqrt();
    let normal = Normal::new(0.0, std_dev).unwrap();

    rng.sample(&normal)
}

impl NeuralNetwork {
    fn new() -> Self {
        let mut rng = rand::thread_rng();
        NeuralNetwork {
            w: [
                [kaiming_init(&mut rng, 2), kaiming_init(&mut rng, 2)],
                [kaiming_init(&mut rng, 2), kaiming_init(&mut rng, 2)],
                [kaiming_init(&mut rng, 2), kaiming_init(&mut rng, 2)],
            ],
            b: [0.0, 0.0, 0.0],
        }
    }

    fn forward(&self, x: [(f64, f64); 4]) -> [ForwardResult; 4] {
        let mut results = [ForwardResult {
            h: [0.0, 0.0],
            y: 0.0,
        }; 4];
        for (i, (x1, x2)) in x.iter().enumerate() {
            let h = [
                sigmoid(self.w[0][0] * x1 + self.w[0][1] * x2 + self.b[0]),
                sigmoid(self.w[1][0] * x1 + self.w[1][1] * x2 + self.b[1]),
            ];
            let y = sigmoid(self.w[2][0] * h[0] + self.w[2][1] * h[1] + self.b[2]);
            results[i] = ForwardResult { h, y };
        }
        results
    }

    fn train(&mut self, x: [(f64, f64); 4], targets: [f64; 4], learning_rate: f64) {
        let results = self.forward(x);

        let mut grad_w = [[0.0; 2]; 3];
        let mut grad_b = [0.0; 3];
        for i in 0..4 {
            let ForwardResult { h, y } = results[i];
            let x = x[i];
            let target = targets[i];

            let error = 2.0 * (y - target);

            let d_y =
                error * sigmoid_derivative(self.w[2][0] * h[0] + self.w[2][1] * h[1] + self.b[2]);

            for j in 0..2 {
                grad_w[2][j] += d_y * h[j];
            }
            grad_b[2] += d_y;

            let d_h = [
                d_y * self.w[2][0]
                    * sigmoid_derivative(self.w[0][0] * x.0 + self.w[0][1] * x.1 + self.b[0]),
                d_y * self.w[2][1]
                    * sigmoid_derivative(self.w[1][0] * x.0 + self.w[1][1] * x.1 + self.b[1]),
            ];

            for j in 0..2 {
                for k in 0..2 {
                    grad_w[j][k] += d_h[j] * x.0;
                    grad_w[j][k] += d_h[j] * x.1;
                }
                grad_b[j] += d_h[j];
            }
        }

        for i in 0..3 {
            for j in 0..2 {
                self.w[i][j] -= learning_rate * grad_w[i][j] / 4.;
            }
            self.b[i] -= learning_rate * grad_b[i] / 4.;
        }
    }
}

fn main() {
    let mut xor_gate = NeuralNetwork::new();
    let data = [(0., 0.), (0., 1.), (1., 0.), (1., 1.)];
    let targets = [0., 1., 1., 0.];
    for _ in 0..100000 {
        xor_gate.train(data, targets, 0.1);
    }
    let results = xor_gate.forward(data);
    for (i, &(x1, x2)) in data.iter().enumerate() {
        println!("Input: ({}, {}), Output: {}", x1, x2, results[i].y);
    }
}
