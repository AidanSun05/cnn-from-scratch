target("format")
    set_kind("headeronly")

    add_includedirs("include", { public = true })
    add_headerfiles("include/format/format.h")
