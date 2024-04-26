use super::differentiable_fn::{DifferentiableFn, Sigmoid};

pub trait LossFn: DifferentiableFn<Args = (f64, f64)> {}

pub struct BCELoss;
impl LossFn for BCELoss {}
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
impl LossFn for BCEWithLogitsLoss {}
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

pub struct MSELoss;
impl LossFn for MSELoss {}
impl DifferentiableFn for MSELoss {
    type Args = (f64, f64);
    fn eval((output, target): Self::Args) -> f64 {
        (output - target).powi(2)
    }
    fn derivative((output, target): Self::Args) -> f64 {
        2. * (output - target)
    }
}
