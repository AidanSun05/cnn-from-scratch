target("inference")
    set_kind("static")
    add_deps("core", "format")

    add_includedirs("include", { public = true })
    add_files("src/nn.c")
