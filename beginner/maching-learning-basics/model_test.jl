# Import Flux (deep learning framework)
using Flux

# Define your model
model = Chain(
    Dense(4, 8, relu),
    Dense(8, 3),
    softmax
)

# Create fake training data
X = rand(Float32, 4, 100)
y = Flux.onehotbatch(rand(1:3, 100), 1:3)

# Define loss and optimizer
loss(x, y) = crossentropy(model(x), y)
opt = ADAM()

# Train model
for epoch in 1:100
    Flux.train!(loss, params(model), [(X, y)], opt)
    println("Epoch $epoch: loss = ", loss(X, y))
end

# Test a prediction
println("Test prediction:", model(rand(Float32, 4)))
