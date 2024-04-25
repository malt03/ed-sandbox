use rand::{rngs::StdRng, Rng, SeedableRng};

const BETA: f64 = 0.8;

const IN: usize = 4;
const ALL: usize = 9;

fn teach_input() -> ([[f64; IN / 2]; IN], [f64; IN]) {
    let g_indata_input = [[0.0, 0.0], [1.0, 0.0], [0.0, 1.0], [1.0, 1.0]];
    let g_indata_tch = [0.0, 1.0, 1.0, 0.0];

    (g_indata_input, g_indata_tch)
}

fn neuro_init<R>(rng: &mut R) -> (Vec<f64>, [[f64; ALL + 2]; ALL - IN])
where
    R: Rng,
{
    let ow: Vec<f64> = (0..ALL + 2)
        .map(|k| if k % 2 == 0 { 1. } else { -1. })
        .collect();

    let mut w_ot_ot = [[0.; ALL + 2]; ALL - IN];

    for k in 0..(ALL - IN) {
        for l in 0..(ALL + 2) {
            w_ot_ot[k][l] = rng.gen::<f64>() * ow[l] * ow[k + IN + 2];
        }
    }

    // dump w_ot_ot
    for k in 0..(ALL - IN) {
        for l in 0..(ALL + 2) {
            print!("{:.2} ", w_ot_ot[k][l]);
        }
        println!();
    }

    (ow, w_ot_ot)
}

fn sigmf(u: f64) -> f64 {
    let u0 = 0.4;
    1. / (1. + (-2. * u / u0).exp())
}

fn neuro_output_calc(
    indata_input: &[f64; IN / 2],
    w_ot_ot: &[[f64; ALL + 2]; ALL - IN],
) -> ([f64; ALL + 2], [f64; ALL - IN]) {
    let mut ot_in = [0.; ALL + 2];
    let mut ot_ot = [0.; ALL - IN];

    ot_in[0] = BETA;
    ot_in[1] = BETA;

    for k in 0..IN {
        ot_in[k + 2] = indata_input[k / 2];
    }
    for _ in 0..2 {
        for k in 0..(ALL - IN) {
            let mut inival = 0.;
            for m in 0..ALL + 2 {
                inival += w_ot_ot[k][m] * ot_in[m];
            }
            ot_ot[k] = sigmf(inival);
        }
        for k in 0..ALL - IN {
            ot_in[k + IN + 2] = ot_ot[k];
        }
    }
    (ot_in, ot_ot)
}

fn neuro_teach_calc(indata_tch: f64, ot_ot: &[f64; ALL - IN]) -> ([f64; 2], f64) {
    let mut del_ot = [0.; 2];

    let wkb = indata_tch - ot_ot[0];

    if wkb > 0. {
        del_ot[0] = wkb;
        del_ot[1] = 0.;
    } else {
        del_ot[0] = 0.;
        del_ot[1] = wkb;
    }

    (del_ot, wkb.abs())
}

const ALPHA: f64 = 0.8;

fn neuro_weight_calc(
    ow: &Vec<f64>,
    w_ot_ot: &mut [[f64; ALL + 2]; ALL - IN],
    ot_in: &[f64; ALL + 2],
    ot_ot: &[f64; ALL - IN],
    del_ot: &[f64; 2],
) {
    for k in 0..ALL - IN {
        let tmp = ot_ot[k].abs() * (1. - ot_ot[k].abs());
        for m in 0..ALL + 2 {
            let del = ALPHA * ot_in[m] * tmp;
            if ow[m] > 0. {
                w_ot_ot[k][m] += del * del_ot[0] * ow[m] * ow[k + IN + 2];
            } else {
                w_ot_ot[k][m] -= del * del_ot[1] * ow[m] * ow[k + IN + 2];
            }
        }
    }
}

fn neuro_calc(
    indata_input: &[f64; IN / 2],
    indata_tch: f64,
    ow: &Vec<f64>,
    w_ot_ot: &mut [[f64; ALL + 2]; ALL - IN],
) -> (f64, [f64; ALL + 2]) {
    let (ot_in, ot_ot) = neuro_output_calc(indata_input, w_ot_ot);
    let (del_ot, err) = neuro_teach_calc(indata_tch, &ot_ot);
    neuro_weight_calc(ow, w_ot_ot, &ot_in, &ot_ot, &del_ot);

    (err, ot_in)
}

fn neuro_output_write(indata_tch: f64, ot_in: &[f64; ALL + 2]) {
    print!("in:");
    for k in 1..=IN / 2 {
        print!("{:.2} ", ot_in[k * 2]);
    }
    print!("-> ");
    print!("{:.5}, {:.2} ", ot_in[IN + 2], indata_tch);
    println!();
}

fn main() {
    let mut rng = StdRng::from_seed([1; 32]);

    let (g_indata_input, g_indata_tch) = teach_input();
    let (ow, mut w_ot_ot) = neuro_init(&mut rng);

    let mut i = 0;
    let mut err = 0.;

    loop {
        i += 1;
        for loopl in 0..IN {
            let (e, ot_in) = neuro_calc(
                &g_indata_input[loopl],
                g_indata_tch[loopl],
                &ow,
                &mut w_ot_ot,
            );
            neuro_output_write(g_indata_tch[loopl], &ot_in);
            err += e;
        }

        println!("err: {}", err);
        if err < 0.1 {
            println!("loop_: {}", i);
            break;
        }
        err = 0.;
    }
}
