#![warn(clippy::pedantic)]

use std::{fs, time::Instant};

use lap_jv::LapJV;
use rand::Rng;
fn main() {
    let size = 5000;
    my_bench(size);
    // lib_bench(size);
}

fn my_bench(size: usize) {
    let mut rng = rand::thread_rng();

    /*  let mut matrix = Vec::with_capacity(size);
     for _ in 0..size {
         let mut row = Vec::with_capacity(size);
         for _ in 0..size {
             row.push(rng.gen::<f64>() * size as f64);
         }
         matrix.push(row);
     }

    let json = serde_json::to_string(&matrix).unwrap();
       println!("{json}");
    */

    let matrix = std::fs::read("./val.json").unwrap();

    let matrix: Vec<Vec<f64>> = serde_json::from_slice(&matrix).unwrap();

    let start = Instant::now();
    let result = LapJV::new(&matrix).solve().unwrap();
    // let result = lap(matrix);
    let elapsed = start.elapsed();
    println!("Calculated in {elapsed:#?}");
    println!("Total cost: {}", result.cost);
}

fn lib_bench(size: usize) {
    let mut rng = rand::thread_rng();

    /*
        let mut vec = Vec::with_capacity(size);
        for _ in 0..size * size {
            vec.push(rng.gen::<f64>() * size as f64);
        }
    */
    let matrix = std::fs::read("./val.json").unwrap();

    let matrix: Vec<Vec<f64>> = serde_json::from_slice(&matrix).unwrap();
    let len = matrix.len();
    let matrix = matrix.into_iter().flatten().collect();

    let matrix = lapjv::Matrix::from_shape_vec((len, len), matrix).unwrap();
    let start = Instant::now();
    let res = lapjv::lapjv(&matrix).unwrap();

    let mut costs = 0.0;
    for i in 0..size {
        let row = res.0[i];
        let col = res.1[i];

        let cost = matrix.row(row)[col];
        costs += cost;
    }

    let elapsed = start.elapsed();
    println!("Calculated in {elapsed:#?} \n total cost: {costs}");
}
