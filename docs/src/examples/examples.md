# Examples

## Polynomial fit

* notebook: [download](visualize_polyfit.ipynb) [nbviewer](@__NBVIEWER_ROOT_URL__/examples/visualize_polyfit.ipynb)
* [script](visualize_polyfit.jl)

```@raw html
<details>
<summary>Model definition</summary>
<p><pre>
```

```@eval
model_file = "../../../../test/test_models/model_polyfit.jl"
res = open(model_file, "r") do io
    join(readlines(io), "\n")
end
res
```

```@raw html
</pre></p>
</details>
```

## FFT Gaussian Process

* notebook: [download](visualize_fft_gp.ipynb) [nbviewer](@__NBVIEWER_ROOT_URL__/examples/visualize_fft_gp.ipynb)
* [script](visualize_fft_gp.jl)

```@raw html
<details>
<summary>Model definition</summary>
<p><pre>
```

```@eval
model_file = "../../../../test/test_models/model_fft_gp.jl"
res = open(model_file, "r") do io
    join(readlines(io), "\n")
end
res
```

```@raw html
</pre></p>
</details>
```
