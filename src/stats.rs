use std::f64::INFINITY;

pub fn mean(data: &Vec<f64>) -> f64 {
    let sum: f64 = data.iter().sum();
    let count = data.len() as f64;
    sum / count
}

pub fn variance(data: &Vec<f64>, mean: f64) -> f64 {
    let sum_of_squares: f64 = data.iter().map(|x| (x - mean).powi(2)).sum();
    let count = data.len() as f64;
    sum_of_squares / count
}

pub fn stdev(data: &Vec<f64>, mean: f64) -> f64 {
    variance(data, mean).sqrt()
}

pub fn max(data: &Vec<f64>) -> f64 {
    data.iter()
        .fold(-INFINITY, |acc, &x| if x > acc { x } else { acc })
}

pub fn min(data: &Vec<f64>) -> f64 {
    data.iter()
        .fold(INFINITY, |acc, &x| if x < acc { x } else { acc })
}
