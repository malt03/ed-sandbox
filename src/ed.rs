use rand::{rngs::ThreadRng, Rng};

const MAX: usize = 32;
const BETA: f64 = 0.8;

fn teach_input(in_: usize, ot: usize, pa: usize) -> ([[f64; MAX]; MAX], [[f64; MAX]; MAX]) {
    let mut g_indata_input = [[0.; MAX]; MAX];
    let mut g_indata_tch = [[0.; MAX]; MAX];

    for k in 0..pa {
        for l in 0..in_ / 2 {
            if k & (1 << l) != 0 {
                g_indata_input[k][l] = 1.;
            } else {
                g_indata_input[k][l] = 0.;
            }
        }
        for n in 0..ot {
            let mut m = 0;
            for l in 0..in_ / 2 {
                if g_indata_input[k][l] > 0.5 {
                    m += 1;
                }
            }
            if m % 2 == 0 {
                g_indata_tch[k][n] = 0.;
            } else {
                g_indata_tch[k][n] = 1.;
            }
        }
    }

    (g_indata_input, g_indata_tch)
}

fn neuro_init(
    rng: &mut ThreadRng,
    in_: usize,
    ot: usize,
    hd: usize,
) -> (
    usize,
    [f64; MAX + 1],
    [[[f64; MAX + 1]; MAX + 1]; MAX + 1],
    [[f64; MAX + 1]; MAX + 1],
) {
    let all = in_ + 1 + hd;

    let mut ow = [0.; MAX + 1];
    let mut w_ot_ot = [[[0.; MAX + 1]; MAX + 1]; MAX + 1];
    let mut ot_in = [[0.; MAX + 1]; MAX + 1];

    for n in 0..ot {
        for k in 0..=(all + 1) {
            ow[k] = (((k as i32 + 1) % 2) * 2 - 1) as f64;
        }
        ow[in_ + 2] = 1.;
        for k in (in_ + 2)..=(all + 1) {
            for l in 0..=(all + 1) {
                if l < 2 || l > 1 || k > all + 1 && l >= in_ + 3 {
                    w_ot_ot[n][k][l] = rng.gen();
                }
                if (k > all + 1 && l < in_ + 2 && l >= 2)
                    || (k != l && k > in_ + 2 && l > in_ + 1)
                    || (k > in_ + 1 && l > in_ + 1 && l < in_ + 3)
                    || (l >= 2 && l < in_ + 2 && k >= in_ + 2 && k < in_ + 3)
                    || (k == l)
                {
                    w_ot_ot[n][k][l] = 0.;
                }
                w_ot_ot[n][k][l] *= ow[l] * ow[k];
            }
        }
        ot_in[n][0] = BETA;
        ot_in[n][1] = BETA;
    }

    (all, ow, w_ot_ot, ot_in)
}

fn sigmf(u: f64) -> f64 {
    let u0 = 0.4;
    1. / (1. + (-2. * u / u0).exp())
}

fn neuro_output_calc(
    in_: usize,
    ot: usize,
    all: usize,
    indata_input: &[f64; MAX],
    w_ot_ot: &[[[f64; MAX + 1]; MAX + 1]; MAX + 1],
    ot_in: &mut [[f64; MAX + 1]; MAX + 1],
) -> [[f64; MAX + 1]; MAX + 1] {
    let mut ot_ot = [[0.; MAX + 1]; MAX + 1];
    for n in 0..ot {
        for k in 2..=in_ + 1 {
            ot_in[n][k] = indata_input[k / 2 - 1];
        }
        for k in in_ + 2..=all + 1 {
            ot_in[n][k] = 0.;
        }
        for _ in 0..2 {
            for k in in_ + 2..=all + 1 {
                let mut inival = 0.;
                for m in 0..=all + 1 {
                    inival += w_ot_ot[n][k][m] * ot_in[n][m];
                }
                ot_ot[n][k] = sigmf(inival);
            }
            for k in in_ + 2..=all + 1 {
                ot_in[n][k] = ot_ot[n][k];
            }
        }
    }
    ot_ot
}

fn neuro_teach_calc(
    in_: usize,
    ot: usize,
    err: &mut f64,
    all: usize,
    indata_tch: &[f64; MAX],
    ot_ot: &[[f64; MAX + 1]; MAX + 1],
) -> [[[f64; 2]; MAX + 1]; MAX + 1] {
    let mut del_ot = [[[0.; 2]; MAX + 1]; MAX + 1];

    for l in 0..ot {
        let wkb = indata_tch[l] - ot_ot[l][in_ + 2];
        *err += wkb.abs();

        if wkb > 0. {
            del_ot[l][in_ + 2][0] = wkb;
            del_ot[l][in_ + 2][1] = 0.;
        } else {
            del_ot[l][in_ + 2][0] = 0.;
            del_ot[l][in_ + 2][1] = -wkb;
        }

        let inival1 = del_ot[l][in_ + 2][0];
        let inival2 = del_ot[l][in_ + 2][1];

        for k in in_ + 3..=all + 1 {
            del_ot[l][k][0] = inival1;
            del_ot[l][k][1] = inival2;
        }
    }
    del_ot
}

const ALPHA: f64 = 0.8;

fn neuro_weight_calc(
    in_: usize,
    ot: usize,
    all: usize,
    ow: &[f64; MAX + 1],
    w_ot_ot: &mut [[[f64; MAX + 1]; MAX + 1]; MAX + 1],
    ot_in: &[[f64; MAX + 1]; MAX + 1],
    ot_ot: &[[f64; MAX + 1]; MAX + 1],
    del_ot: &[[[f64; 2]; MAX + 1]; MAX + 1],
) {
    for n in 0..ot {
        for k in in_ + 2..=all + 1 {
            for m in 0..=all + 1 {
                if w_ot_ot[n][k][m] != 0. {
                    let mut del = ALPHA * ot_in[n][m];
                    del *= ot_ot[n][k].abs();
                    del *= 1. - ot_ot[n][k].abs();
                    if ow[m] > 0. {
                        w_ot_ot[n][k][m] += del * del_ot[n][k][0] * ow[m] * ow[k];
                    } else {
                        w_ot_ot[n][k][m] += del * del_ot[n][k][1] * ow[m] * ow[k];
                    }
                }
            }
        }
    }
}

fn neuro_calc(
    in_: usize,
    ot: usize,
    all: usize,
    err: &mut f64,
    indata_input: &[f64; MAX],
    indata_tch: &[f64; MAX],
    ot_in: &mut [[f64; MAX + 1]; MAX + 1],
    ow: &[f64; MAX + 1],
    w_ot_ot: &mut [[[f64; MAX + 1]; MAX + 1]; MAX + 1],
) -> [[f64; MAX + 1]; MAX + 1] {
    let ot_ot = neuro_output_calc(in_, ot, all, indata_input, &w_ot_ot, ot_in);

    let del_ot = neuro_teach_calc(in_, ot, err, all, indata_tch, &ot_ot);
    neuro_weight_calc(in_, ot, all, ow, w_ot_ot, ot_in, &ot_ot, &del_ot);

    ot_ot
}

fn neuro_output_write(
    in_: usize,
    indata_tch: &[f64; MAX],
    ot_in: &[[f64; MAX + 1]; MAX + 1],
    ot_ot: &[[f64; MAX + 1]; MAX + 1],
) {
    print!("in:");
    for k in 1..=in_ / 2 {
        print!("{:.2} ", ot_in[0][k * 2]);
    }
    print!("-> ");
    print!("{:.5}, {:.2} ", ot_in[0][in_ + 2], indata_tch[0]);
    print!("hd: ");
    for k in in_ + 3..=in_ + 6 {
        if k <= MAX + 1 {
            print!("{:.4} ", ot_ot[0][k]);
        }
    }
    println!();
}

fn main() {
    let mut rng = rand::thread_rng();
    let in_ = 4 * 2;
    let pa = 16;
    let ot = 1;
    let hd = 8;

    let (g_indata_input, g_indata_tch) = teach_input(in_, ot, pa);
    for i in 0..pa {
        println!("{:?}", g_indata_tch[i][0]);
    }
    let (all, ow, mut w_ot_ot, mut ot_in) = neuro_init(&mut rng, in_, ot, hd);

    let mut loop_ = 0;
    let mut err = 0.;
    loop_ += 1;

    loop {
        loop_ += 1;
        for loopl in 0..pa {
            let ot_ot = neuro_calc(
                in_,
                ot,
                all,
                &mut err,
                &g_indata_input[loopl],
                &g_indata_tch[loopl],
                &mut ot_in,
                &ow,
                &mut w_ot_ot,
            );
            neuro_output_write(in_, &g_indata_tch[loopl], &ot_in, &ot_ot);
        }

        println!("err: {}", err);
        if err < 0.1 {
            println!("loop_: {}", loop_);
            break;
        }
        err = 0.;
    }
}
