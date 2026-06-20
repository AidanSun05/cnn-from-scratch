target("example_mnist_train")
    set_default(false)

    add_deps("core", "inference", "training")
    add_files("main.cpp")

target("download_mnist_csv")
    set_default(false)
    set_kind("phony")

    on_build(function (target)
        import("net.http")
        import("utils.archive")

        local train_file_path = path.join(target:targetdir(), "mnist_train.csv.zip")
        http.download("https://github.com/phoebetronic/mnist/raw/refs/heads/main/mnist_train.csv.zip", train_file_path)
        archive.extract(train_file_path, path.join(target:targetdir(), "mnist_train"))

        local test_file_path = path.join(target:targetdir(), "mnist_test.csv.zip")
        http.download("https://github.com/phoebetronic/mnist/raw/refs/heads/main/mnist_test.csv.zip", test_file_path)
        archive.extract(test_file_path, path.join(target:targetdir(), "mnist_test"))
    end)
