add_requires("imgui", { configs = { glfw = true, opengl3 = true } })
add_requires("glfw", { configs = { glfw_include = "system" } })
add_requires("opengl")
add_requireconfs("imgui.glfw", { configs = { glfw_include = "system" } })

target("example_mnist_rt")
    set_default(false)
    add_packages("imgui", "glfw", "opengl")

    -- This may be needed if linker errors related to OpenGL occur
    -- add_ldflags("-lGL")

    add_deps("inference")
    add_files("main.cpp")
