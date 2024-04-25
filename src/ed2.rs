use rand::{rngs::StdRng, Rng, SeedableRng};

fn sigmoid(u: f64) -> f64 {
    let u0 = 0.4;
    1. / (1. + (-2. * u / u0).exp())
}

fn n(i: usize, j: usize) -> f64 {
    (if i % 2 == 0 { 1. } else { -1. }) * (if j % 2 == 0 { 1. } else { -1. })
}

fn neuro_init<R>(rng: &mut R) -> ([[f64; 6]; 8], [f64; 8])
where
    R: Rng,
{
    let mut weight0 = [[0.; 6]; 8];
    let mut weight1 = [0.; 8];

    for i in 0..8 {
        for j in 0..6 {
            weight0[i][j] = rng.gen::<f64>() * n(i, j);
        }
        weight1[i] = rng.gen::<f64>() * n(i, 0);
    }

    (weight0, weight1)
}

const ALPHA: f64 = 0.8;
const BETA: f64 = 0.8;

fn neuro_calc(
    weight0: &mut [[f64; 6]; 8],
    weight1: &mut [f64; 8],
    input: &[f64; 2],
    target: f64,
) -> (f64, f64) {
    let input0 = [BETA, BETA, input[0], input[0], input[1], input[1]];
    let mut input1 = [0.; 8];

    for i in 0..8 {
        for j in 0..6 {
            input1[i] += weight0[i][j] * input0[j];
        }
        input1[i] = sigmoid(input1[i]);
    }

    let mut output = 0.;
    for i in 0..8 {
        output += weight1[i] * input1[i];
    }
    output = sigmoid(output);

    let loss = target - output;

    for i in 0..8 {
        for j in 0..6 {
            let delta = ALPHA * input0[j] * input1[i].abs() * (1. - input1[i].abs());
            if loss > 0. {
                if j % 2 == 0 {
                    weight0[i][j] += delta * loss * n(i, j);
                }
            } else {
                if j % 2 == 1 {
                    weight0[i][j] -= delta * loss * n(i, j);
                }
            }
        }
        let delta = ALPHA * input1[i] * output.abs() * (1. - output.abs());
        if loss > 0. {
            if i % 2 == 0 {
                weight1[i] += delta * loss * n(i, 0);
            }
        } else {
            if i % 2 == 1 {
                weight1[i] -= delta * loss * n(i, 0);
            }
        }
    }

    (output, loss)
}

fn main() {
    let input = [[0., 0.], [0., 1.], [1., 0.], [1., 1.]];
    let targets = [0., 1., 1., 0.];

    let (mut weight0, mut weight1) = neuro_init(&mut StdRng::seed_from_u64(0));

    let mut count = 0;
    loop {
        count += 1;
        let mut err = 0.;
        for i in 0..4 {
            let (output, loss) = neuro_calc(&mut weight0, &mut weight1, &input[i], targets[i]);
            err += loss.abs();
            println!(
                "{}, {} -> {:.5}, {:.2}",
                input[i][0], input[i][1], output, targets[i]
            );
        }
        println!("err: {:.5}", err);
        if err < 0.1 {
            break;
        }
    }
    println!("count: {}", count);
}
