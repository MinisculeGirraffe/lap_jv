#[derive(Debug)]
pub enum LapError {
    NonSquareMatrix,
    NotSolveable,
}

#[derive(Debug, PartialEq, Clone)]
pub struct LapSolution {
    pub row: Vec<usize>,
    pub col: Vec<usize>,
    pub cost: f64,
}

pub struct LapJV<'a> {
    costs: &'a Vec<Vec<f64>>,
    col_sol: Vec<usize>,
    row_sol: Vec<usize>,
    free_rows: Vec<usize>,
    dimension: usize,
    values: Vec<f64>,
    unique: Vec<bool>,
    in_row_not_set: Vec<bool>,
}

impl<'a> LapJV<'a> {
    pub fn new(costs: &'a Vec<Vec<f64>>) -> Self {
        let dimension = costs.len();
        let col_sol = Vec::with_capacity(dimension);
        let row_sol = vec![0; dimension];
        let free_rows = Vec::with_capacity(dimension);
        let values = Vec::with_capacity(dimension);

        let unique = vec![true; dimension];
        let in_row_not_set = vec![true; dimension];
        Self {
            costs,
            col_sol,
            row_sol,
            free_rows,
            dimension,
            values,
            unique,
            in_row_not_set,
        }
    }

    pub fn solve(mut self) -> Result<LapSolution, LapError> {
        for row in self.costs {
            if row.len() != self.dimension {
                return Err(LapError::NonSquareMatrix);
            }
        }

        self.column_reduction();
        self.row_transfer();

        let mut i = 0;

        while !self.free_rows.is_empty() && i < 2 {
            self.augmenting_row_reduction();
            i += 1;
        }

        if !self.free_rows.is_empty() {
            self.augment_solution()?;
        }

        let mut cost = 0.0;
        for i in 0..self.dimension {
            let j = self.col_sol[i];
            cost += self.cost(i, j)
        }

        Ok(LapSolution {
            row: self.row_sol,
            col: self.col_sol,
            cost,
        })
    }

    fn new_vec<T>(&self) -> Vec<T> {
        Vec::with_capacity(self.dimension)
    }
    fn new_filled_vec<T: Clone>(&self, val: T) -> Vec<T> {
        vec![val; self.dimension]
    }

    fn cost(&self, i: usize, j: usize) -> f64 {
        self.costs[i][j]
    }

    fn column_reduction(&mut self) {
        for row in self.costs.iter() {
            // find the lowest value in the column
            let (min_index, min_val) = row.iter().enumerate().skip(1).fold(
                (0, row[0]),
                |(old_idx, old_min), (new_idx, &new_min)| {
                    if new_min < old_min {
                        (new_idx, new_min)
                    } else {
                        (old_idx, old_min)
                    }
                },
            );
            self.col_sol.push(min_index);
            self.values.push(min_val);
        }
    }

    fn row_transfer(&mut self) {
        let mut unique = self.new_filled_vec(true);
        let mut in_row_not_set = self.new_filled_vec(true);

        for j in (0..self.dimension).rev() {
            let i = self.col_sol[j];
            if in_row_not_set[i] {
                self.row_sol[i] = j;
                in_row_not_set[i] = false;
            } else {
                unique[i] = false;
                self.col_sol[j] = usize::MAX;
            }
        }

        for i in 0..self.dimension {
            if in_row_not_set[i] {
                self.free_rows.push(i);
            } else if unique[i] {
                let j = self.row_sol[i];
                let mut min = f64::MAX;
                for j2 in 0..self.dimension {
                    if j2 == j {
                        continue;
                    }
                    let reduced_cost = self.cost(i, j2) - self.values[j];
                    if reduced_cost < min {
                        min = reduced_cost;
                    }
                }
                self.values[j] -= min;
            }
        }
    }

    fn augmenting_row_reduction(&mut self) {
        let mut current = 0;
        let mut rr_cnt = 0;
        let mut new_free_rows = 0;

        let num_free_rows = self.free_rows.len();

        while current < num_free_rows {
            rr_cnt += 1;
            let free_i = self.free_rows[current];
            current += 1;
            // find the first and second minimum reduces cost ofer columns
            let mut u_mins = UMins::find(&self.costs[free_i], &self.values);

            let mut i0 = self.col_sol[u_mins.j1];
            let u_min_new = self.values[u_mins.j1] - (u_mins.u_sub_min - u_mins.u_min);
            let u_min_lowered = u_min_new < self.values[u_mins.j1]; // fixes epsilon bug

            if u_min_lowered {
                // change the reduction of the minimum column to increase the minimum
                // reduced cost in the row to the subminimum.
                self.values[u_mins.j1] = u_min_new;
            } else if i0 != usize::MAX && u_mins.j2.is_some() {
                // minimum and subminimum equal.
                // minimum column j1 is assigned.
                // swap columns j1 and j2, as j2 may be unassigned.
                u_mins.j1 = u_mins.j2.unwrap();
                i0 = self.col_sol[u_mins.j1];
            }

            if i0 != usize::MAX {
                // minimum column j1 assigned earlier.
                if u_min_lowered {
                    // put in current k, and go back to that k.
                    // continue augmenting path i - j1 with i0.
                    current -= 1;
                    self.free_rows[current] = i0;
                } else {
                    // no further augmenting reduction possible.
                    // store i0 in list of free rows for next phase
                    self.free_rows[new_free_rows] = i0;
                    new_free_rows += 1;
                }
            }

            self.row_sol[free_i] = u_mins.j1;
            self.col_sol[u_mins.j1] = free_i;
        }

        self.free_rows.truncate(new_free_rows);
    }

    fn augment_solution(&mut self) -> Result<(), LapError> {
        let mut pred = self.new_filled_vec(0_usize);
        let free_rows = std::mem::take(&mut self.free_rows);
        for row in free_rows {
            let mut i = usize::MAX;
            let mut j = self.shortest_path(row, &mut pred);
            let mut k = 0;
            while i != row {
                i = pred[j];
                self.col_sol[j] = i;
                std::mem::swap(&mut j, &mut self.row_sol[i]);
                k += 1;
                if k > self.dimension {
                    return Err(LapError::NotSolveable);
                }
            }
        }
        Ok(())
    }

    // single iteration od Dijkstra's shortest path with modifications explained in source paper
    // returns the index of the closest free column
    fn shortest_path(&mut self, start_index: usize, pred: &mut [usize]) -> usize {
        let mut column_list = self.new_vec(); // list of columns to be scanned
        let mut cost_distance = self.new_vec(); // cost distance in augmenting path calculation

        let mut lo = 0;
        let mut hi = 0;
        let mut n_ready = 0;

        for (i, pred_ref) in pred.iter_mut().enumerate().take(self.dimension) {
            *pred_ref = start_index;
            column_list.push(i);
            let reduced_cost = self.cost(start_index, i) - self.values[i];
            cost_distance.push(reduced_cost);
        }

        let mut final_j = None;

        while final_j.is_none() {
            if lo == hi {
                n_ready = lo;
                hi = find_dense(self.dimension, lo, &cost_distance, &mut column_list);
                for &j in column_list.iter().take(hi).skip(lo) {
                    if self.col_sol[j] == usize::MAX {
                        final_j = Some(j);
                        break;
                    }
                }
            }

            if final_j.is_none() {
                final_j =
                    self.scan_columns(&mut lo, &mut hi, &mut cost_distance, &mut column_list, pred);
            }
        }

        let min_d = cost_distance[column_list[lo]];
        for &j in column_list.iter().take(n_ready).rev() {
            // update column prices.

            self.values[j] += cost_distance[j] - min_d;
        }
        final_j.unwrap()
    }

    fn scan_columns(
        &self,
        p_lo: &mut usize,
        p_hi: &mut usize,
        cost_distance: &mut [f64],
        column_list: &mut [usize],
        pred: &mut [usize],
    ) -> Option<usize> {
        let mut lo = *p_lo;
        let mut hi = *p_hi;

        while lo != hi {
            let j = column_list[lo];
            lo += 1;
            let i = self.col_sol[j];
            let min_d = cost_distance[j];
            let h = self.cost(i, j) - self.values[j] - min_d;

            //for all columns in todo

            for k in hi..column_list.len() {
                let j = column_list[k];
                let cred_ij = self.cost(i, j) - self.values[j] - h;
                if cred_ij < cost_distance[j] {
                    pred[j] = i;
                    if (cred_ij - min_d).abs() < f64::EPSILON {
                        if self.col_sol[j] == usize::MAX {
                            return Some(j);
                        }
                        column_list[k] = column_list[hi];
                        column_list[hi] = j;
                        hi += 1;
                        break;
                    }
                    cost_distance[j] = cred_ij;
                }
            }
        }
        *p_lo = lo;
        *p_hi = hi;
        None
    }
}

#[inline]
fn find_dense(dim: usize, lo: usize, cost_distance: &[f64], column_list: &mut [usize]) -> usize {
    let mut hi = lo + 1;
    let mut min_d = cost_distance[lo];
    for k in hi..dim {
        let j = column_list[k];
        let h = cost_distance[j];
        if h <= min_d {
            if h < min_d {
                //new minimum found
                hi = lo; // restart list at index low
                min_d = h;
            }
            column_list[k] = column_list[hi];
            column_list[hi] = j;
            hi += 1;
        }
    }
    hi
}
#[derive(Debug, Clone, Copy, PartialEq)]
struct UMins {
    //first minimum value
    u_min: f64,
    // second minimum value
    u_sub_min: f64,
    // first minimum index
    j1: usize,
    // second minimum index
    j2: Option<usize>,
}
impl UMins {
    // finds the minimum and second minimum in a row returing the values and their index
    #[inline]
    fn find(row: &[f64], vals: &[f64]) -> UMins {
        let mut u_min = row[0] - vals[0];
        let mut u_sub_min = f64::MAX;
        let mut j1 = 0;
        let mut j2 = None;

        for j in 1..row.len() {
            let h = row[j] - vals[j];

            if h < u_sub_min {
                if h >= u_min {
                    u_sub_min = h;
                    j2 = Some(j);
                } else {
                    u_sub_min = u_min;
                    u_min = h;
                    j2 = Some(j1);
                    j1 = j;
                }
            }
        }

        UMins {
            u_min,
            u_sub_min,
            j1,
            j2,
        }
    }
}

#[cfg(test)]
mod tests {
    use crate::{LapJV, LapSolution, UMins};

    #[test]
    fn solve_basic() {
        let matrix = vec![
            vec![1.0, 2.0, 3.0],
            vec![4.0, 5.0, 6.0],
            vec![7.0, 8.0, 9.0],
        ];
        let result = LapJV::new(&matrix).solve().unwrap();

        assert_eq!(result.row, vec![1, 2, 0]);
        assert_eq!(result.col, vec![2, 0, 1]);
    }
    #[test]
    fn solve_umins() {
        let row = vec![25.20, 10.37, 30.101, 2.202, 2.192_931, 2.192_932];
        let vals = vec![0.0; row.len()];
        let umins = UMins::find(&row, &vals);
        let correct_umins = UMins {
            u_min: 2.192931,
            u_sub_min: 2.192932,
            j1: 4,
            j2: Some(5),
        };
        assert_eq!(umins, correct_umins);
    }

    // solved in column reduction
    #[test]
    fn solved_column_reduction() {
        let matrix: Vec<Vec<f64>> = vec![
            vec![1000.0, 4.0, 1.0],
            vec![1.0, 1000.0, 3.0],
            vec![5.0, 1.0, 1000.0],
        ];

        let result = LapSolution {
            row: vec![1, 2, 0],
            col: vec![2, 0, 1],
            cost: 3.0,
        };

        assert_eq!(LapJV::new(&matrix).solve().unwrap(), result);
    }

    // Solved in augmenting row reduction.
    #[test]
    fn solved_augmenting_row_reduction() {
        let matrix: Vec<Vec<f64>> = vec![
            vec![5.0, 1000.0, 3.0],
            vec![1000.0, 2.0, 3.0],
            vec![1.0, 5.0, 1000.0],
        ];

        let result = LapSolution {
            row: vec![2, 1, 0],
            col: vec![2, 1, 0],
            cost: 6.0,
        };
        assert_eq!(LapJV::new(&matrix).solve().unwrap(), result);
    }
    #[test]
    // Needs augmentating row reduction - only a single row previously assigned.
    fn needs_augmenting_row_reduction() {
        let matrix: Vec<Vec<f64>> = vec![
            vec![1000.0, 1001.0, 1000.0],
            vec![1000.0, 1000.0, 1001.0],
            vec![1.0, 2.0, 3.0],
        ];

        let result = LapSolution {
            row: vec![2, 1, 0],
            col: vec![2, 1, 0],
            cost: 2001.0,
        };

        assert_eq!(LapJV::new(&matrix).solve().unwrap(), result);
    }
    // Triggers the trackmate bug
    // Solution is ambiguous, [1, 0, 2] gives the same cost, depends on whether
    // in column reduction columns are iterated over from largest to smallest or
    // the other way around.
    #[test]
    fn trackmate_bug() {
        let matrix = vec![
            vec![10.0, 10.0, 13.0],
            vec![4.0, 8.0, 8.0],
            vec![8.0, 5.0, 8.0],
        ];

        let result = LapSolution {
            row: vec![1, 2, 0],
            col: vec![2, 0, 1],
            cost: 13.0 + 4.0 + 5.0,
        };
        assert_eq!(LapJV::new(&matrix).solve().unwrap(), result);
    }
    // This triggered error in augmentation.
}
