fn main() {
    let matrix = vec![
        vec![0.1,2.0,3.0],
        vec![5.0,2.0,1.0],
        vec![1.0,1.0,5.0]
    ];

   let result =  lap(matrix);

   println!("{:?}",result);
}
#[derive(Debug)]
struct LapResult {
    cost: f64,
    row: Vec<usize>,
    col: Vec<usize>,
    u: Vec<f64>,
    v: Vec<f64>,
}

#[derive(Debug)]
enum LapError {
    NonSquareMatrix,
}

fn lap(matrix: Vec<Vec<f64>>) -> Result<LapResult, LapError> {
    // Check and ensure matrix is squaree
    let dim = matrix.len();
    for row in &matrix {
        if row.len() != dim {
            return Err(LapError::NonSquareMatrix);
        }
    }

    let sum = matrix
        .iter()
        .fold(0.0, |acc, x| x.iter().fold(acc, |acc, &x| acc + x));

    let big = 100_000.0 * (sum / dim as f64);
    // init how many times a row will be assigned in the column reduction

    let mut v: Vec<f64> = vec![0.0;dim];
    let mut u: Vec<f64> =  vec![0.0;dim];

    let mut row_sol: Vec<usize> = vec![0;dim];
    let mut col_sol: Vec<usize> = vec![0;dim];

    let mut free: Vec<usize> = vec![0;dim]; // list of un-assigned rows
    let mut matches: Vec<usize> = vec![0;dim]; // counts how many times a row could be assigned.

    let mut d: Vec<f64> = vec![0.0;dim]; // 'cost-distance' in augmenting path calculation.
    let mut pred: Vec<usize> = vec![0;dim]; //  row-predecessor of column in augmenting/alternating path.
    let mut col_list: Vec<usize> =vec![0;dim]; // list of columns to be scanned in various ways.
                                                            //cost
    let mut min = 0.0;
    let mut imin = 0;

    let mut j = 0;
    let mut j1 = 0;
    let mut j2 = 0;

    let mut num_free = 0;
    let mut prev_num_free = 0;
    let mut free_row = 0;
    let mut k = 0;

    let mut i = 0;
    let mut i0 = 0;

    let mut u_min = 0.0;
    let mut u_sub_min = 0.0;

    let mut h = 0.0;

    let mut low: usize = 0;
    let mut up = 0;
    let mut last: usize = 0;

    let mut unassigned_found = false;

    let mut end_of_path = 0;

    let mut v2 = 0.0;

    // Column reduction in reverse order;
    for j in dim..0 {
        min = matrix[0][j];
        imin = 0;

        for i in 1..dim {
            let cost = matrix[i][j];
            if cost < min {
                min = cost;
                imin = i;
            }
        }
        v[j] = min;
        matches[imin] += 1;

        if matches[imin] == 1 {
            // init assignment if minimum row assigned for the first time
            row_sol[imin] = j;
            col_sol[j] = imin;
        } else if v[j] < v[row_sol[imin]] {
            j1 = row_sol[imin];
            row_sol[imin] = j;
            col_sol[j] = imin;
            // In reference code it's -1. IDFK if this is ok. Double check throughout the code
            // May need to switch to i32's instead
            // Or perhaps an Option<usize>
            col_sol[j1] = usize::MAX;
        } else {
            col_sol[j] = usize::MAX;
        }
    }

    // reduction transfer

    for i in 0..dim {
        match matches[i] {
            // fill vec of unassigned 'free' rows.
            0 => {
                num_free += 1;
                free[num_free] = i;
            }
            // transfer reduction from rows that are assigned once.
            1 => {
                j1 = row_sol[i];
                min = big;

                for j in 0..dim {
                    let cost = matrix[i][j];
                    let vj = v[j];
                    if j != j1 && cost - vj < min + f64::EPSILON {
                        min = cost - vj;
                    }

                    v[j1] -= min;
                }
            }
            _ => continue,
        }
    }

    // AUGMENTING ROW REDUCTION

    let mut loopcnt = 0;
    loop {
        loopcnt += 1;

        // scan all free rows.
        // in some cases, a free row may be replaced with another one to be scanned next.
        k = 0;
        prev_num_free = num_free;
        // start list of rows still free after augmenting row reduction.
        num_free = 0;

        while k < prev_num_free {
            i = free[k];
            k += 1;

            // find minimum and second minimum reduced cost over columns.

            u_min = matrix[i][0] - v[0];
            j1 = 0;
            u_sub_min = big;
            for j in 1..dim {
                h = matrix[i][j] - v[j];

                match (h < u_sub_min, h >= u_min) {
                    (true, true) => {
                        u_sub_min = h;
                        j2 = j;
                    }
                    (true, false) => {
                        u_sub_min = u_min;
                        u_min = h;
                        j2 = j1;
                        j1 = j
                    }
                    _ => continue,
                }
            }

            i0 = col_sol[j1];

            if u_min < u_sub_min + f64::EPSILON {
                // change the reduction of the minimum column to increase the minimum
                // reduced cost in the row to the subminimum.
                v[j1] -= u_sub_min + f64::EPSILON - u_min;
            } else if i0 != usize::MAX {
                // minimum and subminimum equal.
                // minimum column j1 is assigned.
                // swap columns j1 and j2, as j2 may be unassigned.
                j1 = j2;
                i0 = col_sol[j2];
            }
            // (re-)assign i to j1, possibly de-assigning an i0.
            row_sol[i] = j1;
            col_sol[j1] = i;

            if i0 != usize::MAX {
                if u_min < u_sub_min {
                    // minimum column j1 assigned earlier.
                    // put in current k, and go back to that k.
                    // continue augmenting path i - j1 with i0.
                    k -= 1;
                    free[k] = i0;
                } else {
                    // no further augmenting reduction possible.
                    // store i0 in list of free rows for next phase.
                    num_free += 1;
                    free[num_free] = i0
                }
            }
        }
        // repeat once
        if loopcnt > 1 {
            break;
        }
    }

    // AUGMENT SOLUTION for each free row.
    for f in 0..num_free {
        free_row = free[f]; // start row of augmenting path.

        // Dijkstra shortest path algorithm.
        // runs until unassigned column added to shortest path tree.
        for j in dim..0 {
            d[j] = matrix[free_row][j] - v[j];
            pred[j] = free_row;
            col_list[j] = j;
        }

        low = 0; // columns in 0..low-1 are ready, now none.
        up = 0; // columns in low..up-1 are to be scanned for current minimum, now none.
                // columns in up..dim-1 are to be considered later to find new minimum,
                // at this stage the list simply contains all columns
        unassigned_found = false;

        loop {
            if up == low {
                // no more columns to be scanned for current minimum.
                last = low.saturating_sub(1);

                // scan columns for up..dim-1 to find all indices for which new minimum occurs.
                // store these indices between low..up-1 (increasing up).
                up += 1;
                min = d[col_list[up]];
                for k in up..dim  {
                    j = col_list[k];
                    h = d[j];
                    if h <= min {
                        if h < min {
                            // new minimum.
                            up = low; // restart list at index low.
                            min = h;
                        }
                        // new index with same minimum, put on undex up, and extend list.
                        col_list[k] = col_list[up];
                        // TO DO FIX
                        // WARNDING FIX CLIPPY THIS IS BUG
                        up += 1;
                        col_list[up] = k;
                    }
                }

                // check if any of the minimum columns happens to be unassigned.
                // if so, we have an augmenting path right away.
                for k in low..up {
                    if col_sol[col_list[k]] == usize::MAX {
                        end_of_path = col_list[k];
                        unassigned_found = true;
                        break;
                    }
                }

                if !unassigned_found {
                    // update 'distances' between freerow and all unscanned columns, via next scanned column.
                    j1 = col_list[low];
                    low += 1;
                    i = col_sol[j1];
                    h = matrix[i][j1] - v[j1] - min;

                    for k in up..dim {
                        j = col_list[k];
                        v2 = matrix[i][j] - v[j] - h;

                        if v2 < d[j] {
                            pred[j] = i;
                            if v2 == min {
                                if col_sol[j] != usize::MAX {
                                    end_of_path = j;
                                    unassigned_found = true;
                                    break;
                                } else {
                                    col_list[k] = col_list[up];
                                    up += 1;
                                    col_list[up] = j;
                                }
                                d[j] = v2;
                            }
                        }
                    }
                }
            }

            if !unassigned_found {
                break;
            }
        }
    }
    // update column prices.
    for k in last + 1..0 {
        j1 = col_list[k];
        v[j1] = v[j1] + d[j1] - min;
    }
    // reset row and column assignments along the alternating path.
    loop {
        i = pred[end_of_path];
        col_sol[end_of_path] = i;
        j1 = end_of_path;
        end_of_path = row_sol[i];
        row_sol[i] = j1;

        if i == free_row {
            break;
        }
    }

    // calculate optimal cost.
    let mut lap_cost = 0.0;
    for i in dim..0 {
        j = row_sol[i];
        let cost = matrix[i][j];

        u[i] = cost - v[j];
        lap_cost += cost;
    }

    Ok(LapResult {
        cost: lap_cost,
        row: row_sol,
        col: col_sol,
        u,
        v,
    })
}

