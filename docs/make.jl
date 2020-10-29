# Use
#
#     DOCUMENTER_DEBUG=true julia --color=yes make.jl local [nonstrict] [fixdoctests]
#
# for local builds.

using Documenter
using MGVI

makedocs(
    sitename = "MGVI",
    modules = [MGVI],
    format = Documenter.HTML(
        prettyurls = !("local" in ARGS),
        canonical = "https://bat.github.io/MGVI.jl/stable/"
    ),
    pages = [
        "Home" => "index.md",
        "API" => "api.md",
        "LICENSE" => "LICENSE.md",
    ],
    doctest = ("fixdoctests" in ARGS) ? :fix : true,
    linkcheck = ("linkcheck" in ARGS),
    strict = !("nonstrict" in ARGS),
)

deploydocs(
    repo = "github.com/bat/MGVI.jl.git",
    forcepush = true,
    push_preview = true,
)
