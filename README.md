# SampleSHAP.jl

Seems to work quite a bit faster than R's [fastshap](https://github.com/bgreenwell/fastshap) package, but it's mostly a playground for me to start learning Julia.

```julia
using CSV, DataFrames, Plots
boston = CSV.read("/Users/bgreenwell/Desktop/trees/data/boston.csv", DataFrame)
X = boston[:, setdiff(names(boston), ["cmedv"])]
xnames = names(X)
X = convert(Matrix, X)
y = convert(Vector, boston.cmedv)

bst = xgboost(X, 200, label = y, eta = 1, max_depth = 6)

# About four times faster than fastshap::explain() in R
shap = explain(bst, X, 1000)
scatter(X[:, 8], shap[:, 8])
scatter(X[:, 15], shap[:, 15])
```
