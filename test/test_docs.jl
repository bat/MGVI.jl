# This file is a part of MGVI.jl, licensed under the MIT License (MIT).

using Test
using MGVI
import Documenter

Documenter.DocMeta.setdocmeta!(
    MGVI,
    :DocTestSetup,
    :(using MGVI);
    recursive=true,
)
Documenter.doctest(MGVI)
