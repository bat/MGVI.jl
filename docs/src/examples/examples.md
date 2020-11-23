# Examples

## Polynomial fit

* notebook: [download](visualize_polyfit.ipynb) [nbviewer](https://nbviewer.jupyter.org/github/bat/MGVInference.jl/blob/gh-pages/dev/examples/visualize_polyfit.ipynb)
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
