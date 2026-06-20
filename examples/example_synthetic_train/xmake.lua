target("example_synthetic_train")
    set_default(false)

    add_deps("core", "inference", "training")
    add_files("main.cpp")

target("example_synthetic_gen")
    set_default(false)
    add_files("gen.c")
