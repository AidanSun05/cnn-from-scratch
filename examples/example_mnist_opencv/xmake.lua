add_requires("opencv")

-- Required for OpenCV
set_exceptions("cxx")

target("example_mnist_opencv")
    set_default(false)
    add_packages("opencv")

    add_deps("core", "inference", "training")
    add_files("main.cpp")
