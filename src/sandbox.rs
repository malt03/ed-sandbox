use rand::Rng;

fn main() {
    let mut rng = rand::thread_rng();
    let x: f64 = rng.gen();
    println!("{}", x);
}
