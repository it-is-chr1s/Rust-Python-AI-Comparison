#[cfg(feature = "cpu")]
mod cpu {
    extern crate blas_src;
    use ndarray::Array2;
    use ndarray_rand::rand_distr::Uniform;
    use ndarray_rand::RandomExt;
    use std::time::Instant;

    pub fn test() {
        println!("CPU / ndarray:");

        let array1 = Array2::<f64>::random((5_000, 5_000), Uniform::new(0., 1.));
        let array2 = Array2::<f64>::random((5_000, 5_000), Uniform::new(0., 1.));

        let start = Instant::now();
        let _cpu_add = &array1 + &array2;
        println!(
            "Time taken for addition: {} s",
            start.elapsed().as_secs_f64()
        );

        let start = Instant::now();
        let _cpu_mul = array1.dot(&array2);
        println!(
            "Time taken for multiplication: {} s",
            start.elapsed().as_secs_f64()
        );
    }
}

#[cfg(feature = "gpu")]
mod gpu {
    extern crate arrayfire;
    use arrayfire::{add, matmul, randu, sync, Dim4, MatProp};
    use std::time::Instant;

    pub fn test() {
        println!("GPU / arrayfire:");

        arrayfire::info();

        let dims = Dim4::new(&[5_000, 5_000, 1, 1]);

        let array1 = randu::<f32>(dims);
        let array2 = randu::<f32>(dims);

        sync(0); // Ensure GPU finishes previous work
        let start = Instant::now();
        let _gpu_add = add(&array1, &array2, true);
        sync(0);
        println!(
            "Time taken for addition: {} s",
            start.elapsed().as_secs_f64()
        );

        sync(0); // Ensure GPU finishes previous work
        let start = Instant::now();
        let _gpu_mul = matmul(&array1, &array2, MatProp::NONE, MatProp::NONE);
        sync(0);
        println!(
            "Time taken for multiplication: {} s",
            start.elapsed().as_secs_f64()
        );
    }
}

fn main() {
    #[cfg(feature = "cpu")]
    cpu::test();

    #[cfg(feature = "gpu")]
    gpu::test();
}
