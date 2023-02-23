# Humpty

[![Build Status](https://github.com/JuliaComputing/Humpty.jl/actions/workflows/CI.yml/badge.svg?branch=main)](https://github.com/JuliaComputing/Humpty.jl/actions/workflows/CI.yml?query=branch%3Amain)
[![Coverage](https://codecov.io/gh/JuliaComputing/Humpty.jl/branch/main/graph/badge.svg)](https://codecov.io/gh/JuliaComputing/Humpty.jl)


## What is this?
This is a package for executing the same instruction on multiple datapoints.
This is in-particular required for batched forward mode AD, where you must pushforward each basis element of the tangent space to get the jacobian.



## Plan
 - The first thing we need is the ability to do batched forward mode AD
    - we need the ability to pushforward a whole basis for the input tangent space, and get out the jacobian of the function, while only running the primal part once
    - One way we can achieve this is by looping over the primal_input paired with each of the basis elements, then have loop-invariant code motion eliminate the duplicate primal computation work (where that is possible, which may not always be the case)
    - An alternative is to overload all linear operators to apply to each basis element within the batch. Only linear operators should be be being done to the tangent part, and there are less of those than you would think. Once again loop-invariant code motion should eliminate the duplicate primal computation work.
 - The second thing we need to do is change this to take advantage of statement level structural spartity information we have
    - Basically we know which basis elements interact with which (unaugmented) primal statements, so we can not emit code which doesn't have the computations which would just perform identity maps
    - This is probably easier done with the explicit looping approach than with the overloading approach, but we will definately need to go in an fiddle with things using a Cassette pass.
 - The the third and final stage is to abstract this away from forwards mode AD and make this a general purpose framework from repeating some work on some data. Such that it is applicable to other things like GPU filling and reverse-mode batching
 - seperately to this we need to check that LLVM is generating good code at each stage


## Representations
We start with the notion of a `Batch`, which is a collection of basis elements.
In the case of the application to AD, a basis element is a [tangent type](https://juliadiff.org/ChainRulesCore.jl/dev/rule_author/tangents.html).

(**TODO**: is `Batch` a good name? Maybe we should give it a more abstracted name like `Seedpod`, since a basis element is a seed.
Does Diffractor already have this as a notion? Is this what Diffractor calls a `Bundle`? We could also call this a `Basis`)


At every stage of pushing forward there is a new `Batch` output.

This in general answers the [question of the what the Jacobian representation for a function taking as input a composite type.](https://github.com/JuliaDiff/ChainRulesCore.jl/issues/66)
It is a `Batch{Tangent{...}}`.
In the case of basis elements that are vectors, this `Batch{Vector{R}}` can be converted into the conventional `Matrix{R}` representation of a Jacobian.

## FAQ

### Why is this called "Humpty.jl"?
Because we are rallying all the kings horses and all the king's men.
IDK, look, I am a programmer, not a poet.

### Is this related to Jax's `vmap`?
Yes, closely.

### Is this related to broadcasting?
It's complicated.
Often broadcasting is exactly the operation we want.
Broadcast any operation done to the `Batch` against all it's basis elements.
