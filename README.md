# SampleSHAP.jl

In an effort to learn Julia, I'm creating a lightweight port of my [fastshap](https://github.com/bgreenwell/fastshap) package for R. It currently only works with [XGBoost.jl](https://github.com/dmlc/XGBoost.jl) models, but the initial results seem promising.

```julia
using CSV, DataFrames, Plots, SampleSHAP, XGBoost

boston = CSV.read("/Users/bgreenwell/Desktop/trees/data/boston.csv", DataFrame)
X = boston[:, setdiff(names(boston), ["cmedv"])]
xnames = names(X)
X = convert(Matrix, X)
y = convert(Vector, boston.cmedv)

bst = xgboost(X, 200, label = y, eta = 1, max_depth = 6)

# About four times faster than fastshap::explain() in R, with probably lots of
# room for improvement (takes about 2--3 seconds per feature on my machine for
# this example)
shap = SampleSHAP.explain(bst, X, 1000)
p1 = scatter(X[:, 8], shap[:, 8], zcolor = X[:, 11])
xlabel!("rm")
p2 = scatter(X[:, 15], shap[:, 15], zcolor = X[:, 11])
xlabel!("lstat")
plot(p1, p2, layout = (1, 2), legend = false, size = (800, 400))
ylabel!("Shapley value")
```
![](plot.png)
