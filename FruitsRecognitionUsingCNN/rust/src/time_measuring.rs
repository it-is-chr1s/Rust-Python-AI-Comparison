use chrono::Local;
use std::fs::OpenOptions;
use std::io::Write;
use std::time::Instant;

pub struct TimeMeasuring {
    time: Instant,
    filename: String,
    log: Vec<String>,
}

impl TimeMeasuring {
    pub fn new(filename: &str) -> Self {
        let time = Instant::now();
        let timestamp = Local::now().format("%Y%m%d-%H%M%S").to_string();
        let filename = format!("{}{}.txt", filename, timestamp);
        let log = Vec::new();

        TimeMeasuring {
            time,
            filename,
            log,
        }
    }

    pub fn took(&mut self, name: &str) {
        let elapsed = self.time.elapsed();
        self.time = Instant::now();
        let elapsed_ms = elapsed.as_secs_f64() * 1000.0;
        let log_entry = format!("{name} needed {:.5} ms", elapsed_ms);

        self.log.push(log_entry.clone());
        println!("{}", log_entry);
    }

    pub fn reset(&mut self) {
        self.time = Instant::now();
    }

    pub fn save_log(&self) {
        let mut file = OpenOptions::new()
            .create(true)
            .append(true)
            .open(&self.filename)
            .expect("Unable to create log file");

        for entry in &self.log {
            writeln!(file, "{}", entry).expect("Unable to write log entry");
        }

        println!("Logs saved to {}", self.filename);
    }
}
