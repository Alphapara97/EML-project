### A Pluto.jl notebook ###
# v0.19.25

using Markdown
using InteractiveUtils


begin
	using Markdown
	using InteractiveUtils
end

begin
	using Random, StatsBase
	using CSV, DataFrames
	using Flux, Evolutionary
	using Plots, UnicodePlots
	using HDF5, LinearAlgebra
	using Distributions
end


using PyCall


md"""
## Final project

**Name: Bharat Jangama & Ansuman Sasmal **

"""


begin
   randno = MersenneTwister(2021)
end


begin
	#dataset link: https://archive.ics.uci.edu/ml/datasets/Diabetic+Retinopathy+Debrecen+Data+Set
	
	print("This dataset contains features extracted from the Messidor image set to predict whether an image contains signs of diabetic retinopathy or not.")
    dataset = DataFrame(CSV.File("messidor_features.csv"))
	
    print("\n\nOriginal dataset size: ", size(dataset))

	#data clean up
	#removing ID column
    dataset = dataset[!, Not(:id)]

	#dropping missing data
    dropmissing!(dataset)
	
    print("\nCleaned dataset size: ", size(dataset))
end


begin
    function plot_class(data)
        positive = sum(data)
        return barplot(
            ["signs of Diabetic Retinopathy", "no signs of Diabetic Retinopathy"],
            [positive, size(data)[1] - positive],
            title="Class distribution",
        )
    end
end


plot_class(dataset.Class)


print("************************** 1. BALANCING DATASET **************************")


begin
	index = (1:nrow(dataset))[isequal.(dataset.Class, 1)]	
	index = sample(randno, index, (2(size(index)[1]) - nrow(dataset)), replace=false)	
	balanced_data = dataset[Not(index), :]
	print("Balanced dataset size: ", size(balanced_data))
end


plot_class(balanced_data.Class)


print("************************ 2. TRAIN TEST SPLITTING ************************")


function train_test_split(data, ratio, stratify)	

	#indexs for positive and negative classes
	positive_idx = (1:nrow(data))[isequal.(stratify, 1)]
	negetive_idx = (1:nrow(data))[isequal.(stratify, 0)]

	#spliting classes
	splitn = round(Int, nrow(data) * ratio * 0.5)
	idx = [
		sample(randno, positive_idx, splitn, replace=false);
		sample(randno, negetive_idx, splitn, replace=false)
	]
	
	#return splits
	return view(data, idx, :), view(data, Not(idx), :)
end


begin
	#splitting dataset into 70% training and 30% testing data	
	train, test = train_test_split(balanced_data, 0.7, balanced_data.Class)
	print("training split size: ", size(train))
	print("\ntesting split size: ", size(test))	
end


begin
	#getting features
	x_train = transpose(Matrix(train[!, Not(:Class)]))
	x_test = transpose(Matrix(test[!, Not(:Class)]))
	
	#getting labels
	y_train = Flux.flatten(train.Class)
	y_test = Flux.flatten(test.Class)
	
	print("training features: ", size(x_train), "  training labels:", size(y_train))
	print("\ntesting features: ", size(x_test), "  testing labels:", size(y_test))	
end


#data scaling
begin
	scaler = fit(ZScoreTransform, x_train)
	StatsBase.transform!(scaler, x_train);
	StatsBase.transform!(scaler, x_test);
end


#generate a randomly initialized fully connected neural network
begin
	#nn_model = Chain(
		#Dense(19, 38, σ),		
		#Dense(38, 19, σ),
		#Dense(19, 1, σ)
	#)
	# nn_model = Chain(
	# 	Dense(19, 128, relu),
    # 	Dense(128, 256, relu),
	# 	Dense(256, 512, relu),
    # 	Dense(256, 128, relu),
    # 	Dense(128, 1, σ)
	# )
	nn_model = Chain(
		Dense(19, 128, relu),
    	Dense(128, 256, relu),
    	Dense(256, 128, relu),
    	Dense(128, 1, σ)
	)
	
	#destructure neural network to build model with new weights
	θ, re = Flux.destructure(nn_model);
end

loss(x, y) = Flux.Losses.binarycrossentropy(x, y)


opt = Flux.setup(Adam(), nn_model)


begin
	X = transpose(Matrix(balanced_data[!, Not(:Class)]))
	Y = Flux.flatten(balanced_data.Class)
	Flux.train!(nn_model, [(X,Y)], opt) do m, x, y
		loss(m(x), y)
	end
end

#function to calculate the accuracy from model predictions
function model_accuracy(nn_model, X, y)
	sum((nn_model(X) .> 0.5) .== y) / size(y)[2]
end


#fitness function
fitness(weights) = -model_accuracy(re(θ.*weights), x_train, y_train)


#training and testing accuracy
begin
	accuracy_train(weights) = model_accuracy(re(weights), x_train, y_train)
	accuracy_test(weights) = model_accuracy(re(weights), x_test, y_test)
	n_iters = 40 #200, 100, 30
end


#Options for the ES
#allows us to store information about each generation
opts = Evolutionary.Options(
	iterations=n_iters,
	show_every=3,
	show_trace=false,
	store_trace=true
)

#crossover function
#uniform selection from the either weight arrays
function uniform_selection(w1::T, w2::T; rng) where {T <: Vector{Float64}}	

	x = rand(Bernoulli(0.1), size(w1)[1])
	w1 = w1 .* x
	w2 = w2 .* x
	res1, res2 = uniform(w1, w2)
	return res1, res1
end


#Function to compute and store the training and testing accuracy of the best individual in each generation
function Evolutionary.trace!(record::Dict{String,Any}, objfun, state, population, method::GA, options) 
	currBest = Evolutionary.minimizer(state)	
	record["training accuracy"] = accuracy_train(currBest)
	record["testing accuracy"] = accuracy_test(currBest)	
end


es_algo = GA(
        selection = rouletteinv, #roulette
        mutation =  gaussian(0.1),
        crossover = uniform_selection,
        mutationRate = 0.95,
        crossoverRate = 0.9,
        populationSize = 2*(19*129),
        ε = 0.02
    )

begin
	# GA algorithm history
	accuracy_trails = Vector()
	train_history = Vector()
	test_history = Vector()
	confusion = Vector()
	for i in 1:10
		#running GA algorithm for weight updates
		history = Evolutionary.optimize(fitness, randn(68609), es_algo, opts)
		
		#predicting results from best model
		preds = re(history.minimizer)(x_test)		

		train_acc = [i.metadata["training accuracy"] for i in history.trace]
		test_acc = [i.metadata["testing accuracy"] for i in history.trace]		
		
		push!(accuracy_trails,  model_accuracy(re(θ.*history.minimizer), x_test, y_test))
					
		
		if size(train_history) == (0,)
			confusion = preds			
			train_history = reshape(train_acc, 1, size(train_acc)[1])
			test_history = reshape(test_acc, 1, size(test_acc)[1])	
		else
			confusion = vcat(confusion, preds)
			train_history = vcat(
				train_history,
				reshape(train_acc, 1, size(train_acc)[1])
			)
			test_history = vcat(
				test_history,
				reshape(test_acc, 1, size(test_acc)[1])
			)			
		end	
		@info i
	end
	train_history = mean(eachrow(train_history))	
	test_history = mean(eachrow(test_history))	
	Dict(
		"train history" => size(train_history),
		"test history" => size(test_history),
	)
end

#average accuracy
@info mean(accuracy_trails)

function confusion_matrix(ŷ, y)
	ŷ = ŷ .> 0.5
	y = vec(y)
	matrix = zeros(2, 2)
	for (a, b) in zip(y, ŷ)
		matrix[a+1, b+1] += 1
	end
	return matrix
end

begin
	matrix = zeros(2, 2)
	for i in eachrow(confusion)
		matrix .+= confusion_matrix(i, y_test)
	end
	matrix ./ 10
end

begin
	plot(1:n_iters+1, train_history, title="Accuracy curve",
			xlabel="epochs",
			ylabel="average accuracy",
			label="Deep; Genetic algorithm: training",
		)
		plot!(1:n_iters+1, test_history, label="Deep; Genetic algorithm: testing")
end

