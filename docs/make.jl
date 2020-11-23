# Use
#
#     DOCUMENTER_DEBUG=true julia --color=yes make.jl local [nonstrict] [fixdoctests]
#
# for local builds.

using Documenter
import Literate
using MGVInference

SRC=joinpath(@__DIR__, "src")

GENERATED = joinpath(@__DIR__, "build")
GENERATED_SRC = joinpath(GENERATED, "src")
GENERATED_DOCS = joinpath(GENERATED, "docs")

EXECUTE_NOTEBOOKS = true

mkdir(GENERATED)
cp(SRC, GENERATED_SRC)

function replace_includes(str)
    included = ["../../../../test/test_models/model_polyfit.jl"]

    # Here the path loads the files from their proper directory,
    # which may not be the directory of the `examples.jl` file!
    path = joinpath(GENERATED_SRC, "examples/")

    for ex in included
        content = read(path*ex, String)
        str = replace(str, "include(\"$(ex)\")" => content)
    end
    return str
end

Literate.notebook(joinpath(GENERATED_SRC, "examples/visualize_polyfit_lit.jl"),
                  joinpath(GENERATED_SRC, "examples/"); name="visualize_polyfit",
                                                        execute=EXECUTE_NOTEBOOKS,
                                                        preprocess=replace_includes)
Literate.script(joinpath(GENERATED_SRC, "examples/visualize_polyfit_lit.jl"),
                joinpath(GENERATED_SRC, "examples/"); name="visualize_polyfit",
                                                      preprocess=replace_includes)

makedocs(
    sitename = "MGVInference",
    modules = [MGVInference],
    root = GENERATED,
    source = "src",
    build = "docs",
    format = Documenter.HTML(
        prettyurls = !("local" in ARGS),
        canonical = "https://bat.github.io/MGVInference.jl/stable/"
    ),
    pages = [
        "Home" => "index.md",
        "Examples" => "examples/examples.md",
        "API" => "api.md",
        "LICENSE" => "LICENSE.md",
    ],
    doctest = ("fixdoctests" in ARGS) ? :fix : true,
    linkcheck = ("linkcheck" in ARGS),
    strict = !("nonstrict" in ARGS),
)

if !("local" in ARGS)
    deploydocs(
        root = GENERATED,
        target = "docs",
        repo = "github.com/bat/MGVInference.jl.git",
        devbranch = "dev",
        forcepush = false,
        push_preview = false
    )
end
