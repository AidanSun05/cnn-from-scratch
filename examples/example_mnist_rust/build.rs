use std::env;
use std::path::PathBuf;

fn main() {
    let profile = env::var("PROFILE").unwrap(); // "debug" or "release"

    let lib_dir = PathBuf::from(format!(
        "../../build/linux/x86_64/{}",
        profile
    ));

    println!("cargo:rustc-link-search=native={}", lib_dir.display());

    // Link libraries
    println!("cargo:rustc-link-lib=static=inference");
    println!("cargo:rustc-link-lib=static=core");
    println!("cargo:rustc-link-lib=gomp");

    let bindings = bindgen::Builder::default()
        .header("../../lib/inference/include/inference/nn.h")
        .parse_callbacks(Box::new(bindgen::CargoCallbacks::new()))
        .generate()
        .expect("Unable to generate bindings");

    let out_path = PathBuf::from(env::var("OUT_DIR").unwrap());
    bindings
        .write_to_file(out_path.join("nn.rs"))
        .expect("Couldn't write bindings!");

    // Re-run if build script changes
    println!("cargo:rerun-if-changed=build.rs");
}
