target("example_cifar10_train")
    set_default(false)

    add_deps("core", "inference", "training")
    add_files("main.cpp")

target("download_cifar10")
    set_default(false)
    set_kind("phony")

    on_build(function (target)
        import("net.http")
        import("utils.archive")

        local file_path = path.join(target:targetdir(), "cifar-10-binary.tar.gz")
        http.download("https://cave.cs.toronto.edu/kriz/cifar-10-binary.tar.gz", file_path)
        archive.extract(file_path, target:targetdir())
    end)
