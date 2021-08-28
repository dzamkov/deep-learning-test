use core::f32;
use std::ops::{Add, AddAssign, Mul};
use std::u32;
use std::{f32::consts::PI, usize};

use image::{GenericImageView, Rgb};
use nalgebra::{dvector, DMatrix};
use rand_distr::Distribution;

const INPUT_SIZE: usize = 32;
const HIDDEN_SIZE: usize = 32;
const NUM_HIDDEN_LAYERS: usize = 4;
const OUTPUT_SIZE: usize = 3;

struct Net {
    first: DMatrix<f32>,
    middle: Vec<DMatrix<f32>>,
    last: DMatrix<f32>,
}

struct Eval {
    layers: Vec<DMatrix<f32>>,
}

impl Net {
    fn new_rand(rng: &mut impl rand::Rng) -> Net {
        let dist =
            rand_distr::Normal::new(0_f32, (1_f32 / ((INPUT_SIZE - 1) as f32)).sqrt()).unwrap();
        let first = DMatrix::from_fn(HIDDEN_SIZE, INPUT_SIZE, |_, _| dist.sample(rng));
        let dist =
            rand_distr::Normal::new(0_f32, (1_f32 / ((HIDDEN_SIZE - 1) as f32)).sqrt()).unwrap();
        let mut middle = Vec::new();
        for _ in 0..(NUM_HIDDEN_LAYERS - 1) {
            middle.push(DMatrix::from_fn(HIDDEN_SIZE, HIDDEN_SIZE, |_, _| {
                dist.sample(rng)
            }));
        }
        let last = DMatrix::from_fn(OUTPUT_SIZE, HIDDEN_SIZE, |_, _| dist.sample(rng));
        return Net {
            first,
            middle,
            last,
        };
    }

    fn zeros() -> Net {
        let first = DMatrix::zeros(HIDDEN_SIZE, INPUT_SIZE);
        let mut middle = Vec::new();
        for _ in 0..(NUM_HIDDEN_LAYERS - 1) {
            middle.push(DMatrix::zeros(HIDDEN_SIZE, HIDDEN_SIZE));
        }
        let last = DMatrix::zeros(OUTPUT_SIZE, HIDDEN_SIZE);
        return Net {
            first,
            middle,
            last,
        };
    }

    fn encode(width: u32, height: u32, samps: &[(u32, u32)]) -> DMatrix<f32> {
        return DMatrix::from_fn(INPUT_SIZE, samps.len(), |i, j| {
            let (x, y) = samps[j];
            let rx = (x as f32) / (width as f32);
            let ry = (y as f32) / (height as f32);
            return match i {
                0 => 1f32,
                1 => rx,
                2 => ry,
                _ => {
                    let n = (i - 3) % 4;
                    let freq = (((i - 3) / 4) as f32) + 1.0;
                    let m = freq * 2.0 * PI;
                    return match n {
                        0 => (rx * m).sin(),
                        1 => (rx * m).cos(),
                        2 => (ry * m).sin(),
                        3 => (ry * m).cos(),
                        _ => panic!(),
                    };
                }
            };
        });
    }

    fn activation(x: f32) -> f32 {
        if x > 0_f32 {
            return x;
        } else {
            return x * (1f32 / 1024f32);
        }
    }

    fn d_activation(x: f32, y: f32) -> f32 {
        if x > 0_f32 {
            return y;
        } else {
            return y * (1f32 / 1024f32);
        }
    }

    fn eval(&self, input: DMatrix<f32>) -> Eval {
        let mut layers = Vec::new();
        let mut hidden = &self.first * &input;
        layers.push(input);
        hidden = hidden.map(Net::activation);
        hidden[0] = 1f32;
        for i in 0..(NUM_HIDDEN_LAYERS - 1) {
            let mut next = &self.middle[i] * &hidden;
            next = next.map(Net::activation);
            next[0] = 1f32;
            layers.push(hidden);
            hidden = next;
        }
        let res = &self.last * &hidden;
        layers.push(hidden);
        layers.push(res);
        return Eval { layers };
    }

    fn grad(&self, eval: &Eval, d_output: &DMatrix<f32>) -> Net {
        let prev = &eval.layers[eval.layers.len() - 2];
        let last = d_output * prev.transpose();
        let d_next = self.last.tr_mul(d_output);
        let mut d_next = prev.zip_map(&d_next, Net::d_activation);
        let mut middle = Vec::new();
        for i in (0..(NUM_HIDDEN_LAYERS - 1)).rev() {
            let prev = &eval.layers[1 + i];
            middle.push(&d_next * prev.transpose());
            d_next = self.middle[i].tr_mul(&d_next);
            d_next = prev.zip_map(&d_next, Net::d_activation);
        }
        middle.reverse();
        let prev = &eval.layers[0];
        let first = d_next * prev.transpose();
        return Net {
            first,
            middle,
            last,
        };
    }

    fn d_output(target: &DMatrix<f32>, res: &DMatrix<f32>) -> DMatrix<f32> {
        return res - target;
    }

    fn cost(target: &DMatrix<f32>, res: &DMatrix<f32>) -> f32 {
        return (res - target).map(|x| x * x).sum();
    }

    fn point_sqr(&self) -> Net {
        let first = self.first.map(|x| x * x);
        let middle = self.middle.iter().map(|a| a.map(|x| x * x)).collect();
        let last = self.last.map(|x| x * x);
        return Net {
            first,
            middle,
            last,
        };
    }

    fn update(&mut self, rate: f32, tol: f32, vel: &Net, var: &Net) {
        let update = |x: &mut f32, y: f32, z: f32| *x -= rate * y / (z.sqrt() + tol);
        self.first.zip_zip_apply(&vel.first, &var.first, update);
        for i in 0..self.middle.len() {
            self.middle[i].zip_zip_apply(&vel.middle[i], &var.middle[i], update);
        }
        self.last.zip_zip_apply(&vel.last, &var.last, update);
    }
}

impl Add for &Net {
    type Output = Net;

    fn add(self, rhs: &Net) -> Net {
        let first = &self.first + &rhs.first;
        let middle = self
            .middle
            .iter()
            .zip(rhs.middle.iter())
            .map(|(a, b)| a + b)
            .collect();
        let last = &self.last + &rhs.last;
        return Net {
            first,
            middle,
            last,
        };
    }
}

impl AddAssign<&Net> for Net {
    fn add_assign(&mut self, rhs: &Net) {
        self.first += &rhs.first;
        for i in 0..self.middle.len() {
            self.middle[i] += &rhs.middle[i];
        }
        self.last += &rhs.last;
    }
}

impl Mul<f32> for &Net {
    type Output = Net;

    fn mul(self, rhs: f32) -> Net {
        let first = &self.first * rhs;
        let middle = self.middle.iter().map(|a| a * rhs).collect();
        let last = &self.last * rhs;
        return Net {
            first,
            middle,
            last,
        };
    }
}

impl Eval {
    fn res(&self) -> &DMatrix<f32> {
        return &self.layers.last().unwrap();
    }
}

fn main() {
    let im = image::open("data/photo.jpg").unwrap();
    let width = im.width();
    let height = im.height();

    // Build sampling sets
    let ln_block_size = 3u32;
    let block_size = 1u32 << ln_block_size;
    let mut samps = Vec::new();
    for o in 0..(block_size * block_size) {
        let mut phase_samps = Vec::new();
        let seq = |n: u32| {
            let n = (n << 1) | (n >> 2);
            let n = 0x33333333 & n;
            let n = (n << 2) | (n >> 4);
            let n = 0x0F0F0F0F & n;
            let n = (n << 4) | (n >> 8);
            let n = 0x00FF00FF & n;
            let n = (n << 8) | (n >> 16);
            return (n & 0x0000FFFF) >> (16 - ln_block_size);
        };
        let ox = seq(0x55555555 & o);
        let oy = seq((0xAAAAAAAA & o) >> 1);
        for x in 0..(width >> ln_block_size) {
            for y in 0..(height >> ln_block_size) {
                phase_samps.push(((x << ln_block_size) + ox, (y << ln_block_size) + oy));
            }
        }
        samps.push(phase_samps);
    }
    let mut full_samps = Vec::new();
    for y in 0..height {
        for x in 0..width {
            full_samps.push((x, y));
        }
    }

    // Load target data
    let cim = im.into_rgb8();
    let mut targets = Vec::new();
    for phase_samps in &samps {
        let mut target = DMatrix::zeros(OUTPUT_SIZE, phase_samps.len());
        for i in 0..phase_samps.len() {
            let (x, y) = phase_samps[i];
            let Rgb([r, g, b]) = *cim.get_pixel(x, y);
            target.set_column(i as usize,
                &dvector![(r as f32) / 255.0, (g as f32) / 255.0, (b as f32) / 255.0],
            );
        }
        targets.push(target);
    }
    let mut full_target = DMatrix::zeros(OUTPUT_SIZE, full_samps.len());
    for i in 0..full_samps.len() {
        let (x, y) = full_samps[i];
        let Rgb([r, g, b]) = *cim.get_pixel(x, y);
        full_target.set_column(i as usize,
            &dvector![(r as f32) / 255.0, (g as f32) / 255.0, (b as f32) / 255.0],
        );
    }

    // Construct encoded inputs
    let mut inputs = Vec::new();
    for phase_samps in &samps {
        inputs.push(Net::encode(width, height, &phase_samps));
    }
    let full_input = Net::encode(width, height, &full_samps);

    // Reconstruct from neural net
    let vel_decay = 0.9;
    let var_decay = 0.9;
    let tol = 0.0001;
    let mut rng = rand::thread_rng();
    let mut net = Net::new_rand(&mut rng);
    let mut vel = Net::zeros();
    let mut var = Net::zeros();
    let mut i = 0;
    loop {
        i = i + 1;
        let eval = net.eval(inputs[i % samps.len()].clone());
        let target = &targets[i % samps.len()];
        let d_output = Net::d_output(target, eval.res());

        let grad = net.grad(&eval, &d_output);
        vel = &(&vel * vel_decay) + &(&grad * (1f32 - vel_decay));
        var = &(&var * var_decay) + &(&Net::point_sqr(&grad) * (1f32 - var_decay));

        let rate = 0.01 / (1.0 + 0.001 * (i as f32));
        net.update(
            rate,
            tol,
            &(&vel * (1.0 / (1.0 - vel_decay.powf(i as f32)))),
            &(&var * (1.0 / (1.0 - var_decay.powf(i as f32)))),
        );

        if i % 100 == 0 {
            let eval = net.eval(full_input.clone());
            let mut oim = image::ImageBuffer::new(width, height);
            for (x, y, pixel) in oim.enumerate_pixels_mut() {
                let val = eval.res().column((x + y * width) as usize);
                let r = val[0];
                let g = val[1];
                let b = val[2];
                *pixel = Rgb([(r * 255.0) as u8, (g * 255.0) as u8, (b * 255.0) as u8]);
            }
            println!("Round {} cost {}", i, Net::cost(&full_target, eval.res()));
            oim.save(format!("out/{}.png", i)).unwrap();
        }
    }
}
