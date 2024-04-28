use super::differentiable_fn::{DifferentiableFn, Sigmoid};

pub struct BCELoss;
impl DifferentiableFn for BCELoss {
    type Args = (f64, f64);
    fn eval((output, target): Self::Args) -> f64 {
        let output = output.max(1e-12).min(1. - 1e-12);
        -(target * output.ln() + (1. - target) * (1. - output).ln())
    }
    fn derivative((output, target): Self::Args) -> f64 {
        let output = output.max(1e-12).min(1. - 1e-12);
        -(target / output - (1.0 - target) / (1.0 - output))
    }
}

pub struct BCEWithLogitsLoss;
impl DifferentiableFn for BCEWithLogitsLoss {
    type Args = (f64, f64);
    fn eval((output, target): Self::Args) -> f64 {
        let output = Sigmoid::eval(output);
        BCELoss::eval((output, target))
    }
    fn derivative((output, target): Self::Args) -> f64 {
        Sigmoid::eval(output) - target
    }
}

pub struct CrossEntropyLoss;
impl CrossEntropyLoss {
    pub fn softmax(input: &Vec<f64>) -> Vec<f64> {
        let max = input.iter().cloned().fold(f64::NEG_INFINITY, f64::max);
        let exps: Vec<f64> = input.iter().map(|x| (x - max).exp()).collect();
        let sum: f64 = exps.iter().sum();
        exps.iter().map(|x| x / sum).collect()
    }

    pub fn eval((output, target): (&Vec<f64>, &Vec<f64>)) -> f64 {
        let output = CrossEntropyLoss::softmax(output);
        -target
            .iter()
            .zip(output.iter())
            .map(|(t, o)| *t * o.ln())
            .sum::<f64>()
    }

    pub fn derivative((output, target): (&Vec<f64>, &Vec<f64>)) -> Vec<f64> {
        let output = CrossEntropyLoss::softmax(output);
        output
            .iter()
            .zip(target.iter())
            .map(|(o, t)| o - t)
            .collect()
    }
}

pub struct MSELoss;
impl DifferentiableFn for MSELoss {
    type Args = (f64, f64);
    fn eval((output, target): Self::Args) -> f64 {
        (output - target).powi(2)
    }
    fn derivative((output, target): Self::Args) -> f64 {
        2. * (output - target)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_softmax() {
        let input = vec![1.0, 2.0, 3.0];
        let output = CrossEntropyLoss::softmax(&input);
        assert_eq!(
            output,
            vec![0.09003057317038046, 0.24472847105479764, 0.6652409557748218]
        );
    }

    #[test]
    fn test_cross_entropy_loss() {
        let output = vec![1.0, 2.0, 3.0];
        let target = vec![0.0, 1.0, 0.0];
        let loss = CrossEntropyLoss::eval((&output, &target));
        assert_eq!(loss, 1.4076059644443804);
    }

    #[test]
    fn test_cross_entropy_loss_derivative() {
        let output = vec![1.0, 2.0, 3.0];
        let target = vec![0.0, 1.0, 0.0];
        let derivative = CrossEntropyLoss::derivative((&output, &target));
        assert_eq!(
            derivative,
            vec![0.09003057317038046, -0.7552715289452023, 0.6652409557748218]
        );
    }
}
