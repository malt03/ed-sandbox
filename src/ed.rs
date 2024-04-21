use rand::{rngs::StdRng, Rng, SeedableRng};

const MAX: usize = 32;
const BETA: f64 = 0.8;

const PA: usize = 16;
const IN: usize = 8;

fn teach_input() -> ([[f64; IN / 2]; PA], Vec<f64>) {
    let g_indata_input = [
        [0.0, 0.0, 0.0, 0.0], // 0.0,
        [1.0, 0.0, 0.0, 0.0], // 1.0,
        [0.0, 1.0, 0.0, 0.0], // 1.0,
        [1.0, 1.0, 0.0, 0.0], // 0.0,
        [0.0, 0.0, 1.0, 0.0], // 1.0,
        [1.0, 0.0, 1.0, 0.0], // 0.0,
        [0.0, 1.0, 1.0, 0.0], // 0.0,
        [1.0, 1.0, 1.0, 0.0], // 1.0,
        [0.0, 0.0, 0.0, 1.0], // 1.0,
        [1.0, 0.0, 0.0, 1.0], // 0.0,
        [0.0, 1.0, 0.0, 1.0], // 0.0,
        [1.0, 1.0, 0.0, 1.0], // 1.0,
        [0.0, 0.0, 1.0, 1.0], // 0.0,
        [1.0, 0.0, 1.0, 1.0], // 1.0,
        [0.0, 1.0, 1.0, 1.0], // 1.0,
        [1.0, 1.0, 1.0, 1.0], // 0.0
    ];

    let mut g_indata_tch = vec![0.; PA];

    for k in 0..PA {
        let mut c = 0;
        for l in 0..IN / 2 {
            if g_indata_input[k][l] == 1. {
                c += 1;
            }
        }

        g_indata_tch[k] = if c % 2 == 0 { 0. } else { 1. };
    }

    println!("{:?}", g_indata_tch);

    (g_indata_input, g_indata_tch)
}

fn neuro_init<R>(
    rng: &mut R,
    hd: usize,
) -> (
    usize,
    [f64; MAX + 1],
    [[f64; MAX + 1]; MAX + 1],
    [f64; MAX + 1],
)
where
    R: Rng,
{
    let all = IN + 1 + hd;

    let mut ow = [0.; MAX + 1];
    let mut w_ot_ot = [[0.; MAX + 1]; MAX + 1];
    let mut ot_in = [0.; MAX + 1];

    for k in 0..=(all + 1) {
        ow[k] = (((k as i32 + 1) % 2) * 2 - 1) as f64;
    }
    ow[IN + 2] = 1.;
    for k in (IN + 2)..=(all + 1) {
        for l in 0..=(all + 1) {
            if l < 2 || l > 1 || k > all + 1 && l >= IN + 3 {
                w_ot_ot[k][l] = rng.gen();
            }
            if (k > all + 1 && l < IN + 2 && l >= 2)
                || (k != l && k > IN + 2 && l > IN + 1)
                || (k > IN + 1 && l > IN + 1 && l < IN + 3)
                || (l >= 2 && l < IN + 2 && k >= IN + 2 && k < IN + 3)
                || (k == l)
            {
                w_ot_ot[k][l] = 0.;
            }
            w_ot_ot[k][l] *= ow[l] * ow[k];
        }
    }
    ot_in[0] = BETA;
    ot_in[1] = BETA;

    (all, ow, w_ot_ot, ot_in)
}

fn sigmf(u: f64) -> f64 {
    let u0 = 0.4;
    1. / (1. + (-2. * u / u0).exp())
}

fn neuro_output_calc(
    all: usize,
    indata_input: &[f64; 4],
    w_ot_ot: &[[f64; MAX + 1]; MAX + 1],
    ot_in: &mut [f64; MAX + 1],
) -> [f64; MAX + 1] {
    let mut ot_ot = [0.; MAX + 1];
    for k in 2..=IN + 1 {
        ot_in[k] = indata_input[k / 2 - 1];
    }
    for k in IN + 2..=all + 1 {
        ot_in[k] = 0.;
    }
    for _ in 0..2 {
        for k in IN + 2..=all + 1 {
            let mut inival = 0.;
            for m in 0..=all + 1 {
                inival += w_ot_ot[k][m] * ot_in[m];
            }
            ot_ot[k] = sigmf(inival);
        }
        for k in IN + 2..=all + 1 {
            ot_in[k] = ot_ot[k];
        }
    }
    ot_ot
}

fn neuro_teach_calc(
    err: &mut f64,
    all: usize,
    indata_tch: f64,
    ot_ot: &[f64; MAX + 1],
) -> [[f64; MAX + 1]; MAX + 1] {
    let mut del_ot = [[0.; MAX + 1]; MAX + 1];

    let wkb = indata_tch - ot_ot[IN + 2];
    *err += wkb.abs();

    if wkb > 0. {
        del_ot[IN + 2][0] = wkb;
        del_ot[IN + 2][1] = 0.;
    } else {
        del_ot[IN + 2][0] = 0.;
        del_ot[IN + 2][1] = -wkb;
    }

    let inival1 = del_ot[IN + 2][0];
    let inival2 = del_ot[IN + 2][1];

    for k in IN + 3..=all + 1 {
        del_ot[k][0] = inival1;
        del_ot[k][1] = inival2;
    }

    del_ot
}

const ALPHA: f64 = 0.8;

fn neuro_weight_calc(
    all: usize,
    ow: &[f64; MAX + 1],
    w_ot_ot: &mut [[f64; MAX + 1]; MAX + 1],
    ot_in: &[f64; MAX + 1],
    ot_ot: &[f64; MAX + 1],
    del_ot: &[[f64; MAX + 1]; MAX + 1],
) {
    for k in IN + 2..=all + 1 {
        for m in 0..=all + 1 {
            if w_ot_ot[k][m] != 0. {
                let mut del = ALPHA * ot_in[m];
                del *= ot_ot[k].abs();
                del *= 1. - ot_ot[k].abs();
                if ow[m] > 0. {
                    w_ot_ot[k][m] += del * del_ot[k][0] * ow[m] * ow[k];
                } else {
                    w_ot_ot[k][m] += del * del_ot[k][1] * ow[m] * ow[k];
                }
            }
        }
    }
}

fn neuro_calc(
    all: usize,
    err: &mut f64,
    indata_input: &[f64; 4],
    indata_tch: f64,
    ot_in: &mut [f64; MAX + 1],
    ow: &[f64; MAX + 1],
    w_ot_ot: &mut [[f64; MAX + 1]; MAX + 1],
) -> [f64; MAX + 1] {
    let ot_ot = neuro_output_calc(all, indata_input, w_ot_ot, ot_in);

    let del_ot = neuro_teach_calc(err, all, indata_tch, &ot_ot);
    neuro_weight_calc(all, ow, w_ot_ot, ot_in, &ot_ot, &del_ot);

    ot_ot
}

fn neuro_output_write(indata_tch: f64, ot_in: &[f64; MAX + 1], ot_ot: &[f64; MAX + 1]) {
    print!("in:");
    for k in 1..=IN / 2 {
        print!("{:.2} ", ot_in[k * 2]);
    }
    print!("-> ");
    print!("{:.5}, {:.2} ", ot_in[IN + 2], indata_tch);
    print!("hd: ");
    for k in IN + 3..=IN + 6 {
        if k <= MAX + 1 {
            print!("{:.4} ", ot_ot[k]);
        }
    }
    println!();
}

fn main() {
    let mut rng = StdRng::from_seed([1; 32]);
    let hd = 8;

    let (g_indata_input, g_indata_tch) = teach_input();
    let (all, ow, mut w_ot_ot, mut ot_in) = neuro_init(&mut rng, hd);

    let mut loop_ = 0;
    let mut err = 0.;
    loop_ += 1;

    loop {
        loop_ += 1;
        for loopl in 0..PA {
            let ot_ot = neuro_calc(
                all,
                &mut err,
                &g_indata_input[loopl],
                g_indata_tch[loopl],
                &mut ot_in,
                &ow,
                &mut w_ot_ot,
            );
            neuro_output_write(g_indata_tch[loopl], &ot_in, &ot_ot);
        }

        println!("err: {}", err);
        if err < 0.1 {
            println!("loop_: {}", loop_);
            break;
        }
        err = 0.;
    }
}
