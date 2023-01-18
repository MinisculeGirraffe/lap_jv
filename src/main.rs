#![warn(clippy::pedantic)]

use std::{fs, time::Instant};

use lap_jv::LapJV;

fn main() {
    let matrix = vec![
        vec![10.0, 10.0, 13.0],
        vec![4.0, 8.0, 8.0],
        vec![8.0, 5.0, 8.0],
    ];
    let result = LapJV::new(&matrix).solve().unwrap();

    println!("Result: {:?}", result);
}
/*
fn bench() {
    let size = 1000;
    let mut rng = rand::thread_rng();
    let mut matrix = Vec::with_capacity(size);
    for _ in 0..size {
        let mut row = Vec::with_capacity(size);
        for _ in 0..size {
            row.push(rng.gen::<f64>() * size as f64);
        }
        matrix.push(row);
    }

    my_bench(size, matrix.clone());
    lib_bench(size, matrix.clone());
}

fn my_bench(size: usize, matrix: Vec<Vec<f64>>) {
    let start = Instant::now();
    let result = LapJV::new(&matrix).solve().unwrap();
    // let result = lap(matrix);
    let elapsed = start.elapsed();
    println!("Calculated in {elapsed:#?} - Total Cost: {}", result.cost);
}

fn lib_bench(size: usize, matrix: Vec<Vec<f64>>) {
    let vec = matrix.into_iter().flatten().collect::<Vec<f64>>();

    let matrix = lapjv::Matrix::from_shape_vec((size, size), vec).unwrap();
    let start = Instant::now();
    let res = lapjv::lapjv(&matrix).unwrap();

    let elapsed = start.elapsed();
    let mut costs = 0.0;
    for i in 0..size {
        let row = res.0[i];
        let col = res.1[i];

        let cost = matrix.row(row)[col];
        costs += cost;
    }

    println!("Calculated in {elapsed:#?} - Total Cost: {costs}");
}


*/
