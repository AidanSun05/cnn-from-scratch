add_requires("openmp")

target("training")
    set_kind("static")
    add_packages("openmp")
    add_deps("core", "format")

    add_includedirs("include", { public = true })
    add_files("src/matrix.cpp", "src/nn.cpp", "src/optimizer.cpp")
