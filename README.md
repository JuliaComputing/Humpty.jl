# Humpty

[![Build Status](https://github.com/JuliaComputing/Humpty.jl/actions/workflows/CI.yml/badge.svg?branch=main)](https://github.com/JuliaComputing/Humpty.jl/actions/workflows/CI.yml?query=branch%3Amain)
[![Coverage](https://codecov.io/gh/JuliaComputing/Humpty.jl/branch/main/graph/badge.svg)](https://codecov.io/gh/JuliaComputing/Humpty.jl)


## What is this?
This is a package for executing the same instruction on multiple datapoints.
This is in-particular required for batched forward mode AD, where you must pushforward each basis element of the tangent space to get the jacobian.

## Representations
We start with the notion of a `Batch`, which is a collection of basis elements.
In the case of the application to AD, a basis element is a [tangent type](https://juliadiff.org/ChainRulesCore.jl/dev/rule_author/tangents.html).


At every stage of pushing forward there is a new `Batch` output.

This in general answers the [question of the what the Jacobian representation for a function taking as input a composite type.](https://github.com/JuliaDiff/ChainRulesCore.jl/issues/66)
It is a `Batch{Tangent{...}}`.
In the case of basis elements that are vectors, this `Batch{Vector{R}}` can be converted into the conventional `Matrix{R}` representation of a Jacobian.

## FAQ

### Why is this called "Humpty.jl"?
Because we are rallying all the kings horses and all the king's men.
IDK, look, I am a programmer, not a poet.

### Is this related to broadcasting?
It's complicated.
Often broadcasting is exactly the operation we want.
Broadcast any operation done to the `Batch` against all it's basis elements.

### Since `frule` is fused primal computation with pushforward how do you avoid doing the primal computation many times?

We don't, we just make sure the compiler removes the unneeded work.
Which it currently does for primals returning scalars but not for arrays due to [JuliaLang/Julia#48808](https://github.com/JuliaLang/julia/issues/48808)
