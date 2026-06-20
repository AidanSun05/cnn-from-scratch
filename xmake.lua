add_rules("mode.debug", "mode.release", "mode.releasedbg")
set_languages("c++20", "c11")

add_vectorexts("avx2", "fma")

if is_mode("release", "releasedbg") then
    set_exceptions("no-cxx")
    set_optimize("fastest") -- -O3
    set_fpmodels("fast") -- -ffast-math
    add_cxflags("-fno-plt", "-funroll-loops", "-fno-math-errno", "-fno-trapping-math", "-march=native", "-flto")
end

includes("core", "examples", "lib", "test")
