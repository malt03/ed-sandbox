use std::f64::consts::E;

pub trait DifferentiableFn {
    type Args;
    fn eval(input: Self::Args) -> f64;
    fn derivative(input: Self::Args) -> f64;
}

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
