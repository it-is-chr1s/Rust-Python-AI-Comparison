use std::env;

fn main() {
    let lib_path = env::var("LIBRARY_PATH").unwrap_or_else(|_| "C:\\path\\to\\libs".to_string());

    println!("cargo:rustc-link-search=native={}", lib_path);
    println!("cargo:rustc-link-lib=dylib=openblas");
}
