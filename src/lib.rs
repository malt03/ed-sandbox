mod ed3;
pub mod mnist;

pub use ed3::gate::Gate;
pub use ed3::layer::Layer;
pub use ed3::mnist::Mnist;
pub use ed3::{differentiable_fn::*, loss_fn::*};
pub use mnist as dataset;
