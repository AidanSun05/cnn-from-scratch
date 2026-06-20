add_requires("openmp")

target("core")
    set_kind("static")

    add_packages("openmp")
    add_includedirs("include", { public = true })
    add_files("src/conv.c", "src/gemm.c", "src/ops.c")
