use std::f64::consts::E;

pub trait DifferentiableFn: Send {
    type Args;
    fn eval(input: Self::Args) -> f64;
    fn derivative(input: Self::Args) -> f64;
}

#[derive(Debug)]
pub struct PassThrough;
impl DifferentiableFn for PassThrough {
    type Args = f64;
    fn eval(input: Self::Args) -> f64 {
        input
    }
    fn derivative(_: Self::Args) -> f64 {
        1.0
    }
}

#[derive(Debug)]
pub struct Sigmoid;
impl DifferentiableFn for Sigmoid {
    type Args = f64;
    fn eval(input: Self::Args) -> f64 {
        1.0 / (1.0 + E.powf(-input))
    }
    fn derivative(input: Self::Args) -> f64 {
        let s = Sigmoid::eval(input);
        s * (1.0 - s)
    }
}
