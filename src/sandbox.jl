using Plots

x = -5.0:0.01:5.0
y = pdf(TDist(3.0), x)
N = 10000
plot(x,y)
histogram(samples)
