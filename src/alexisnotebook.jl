# A replica of Alexis Cook tutorial on the Titanic challenge


## Load packages
using CSV
using DataFrames
using DecisionTree


## Load data
train_data = DataFrame(CSV.File("input/train.csv"))
show(train_data, allcols=true)

test_data = DataFrame(CSV.File("input/test.csv"))

## find how many women survived

rate_women = (subset(train_data, :Sex => ByRow(==("female")))[!, :Survived] |> sum) / ((subset(train_data, :Sex => ByRow(==("female")))|>size)[1])
print("% of women that survived: $rate_women")

## find how many men survived
rate_men = (subset(train_data, :Sex => ByRow(==("male")))[!, :Survived] |> sum) / ((subset(train_data, :Sex => ByRow(==("male")))|>size)[1])
print("% of men that survived: $rate_men")

## Random Forest Model

y = train_data[!, :Survived]

features = [:Pclass, :Sex, :SibSp, :Parch]
X = train_data[!, features]

model = RandomForestClassifier(n_trees=100, max_depth=5)

fit!(model, Matrix(X), y)

## Predictions

predictions = predict(model, Matrix(test_data[!, features]))

submission = DataFrame(PassengerId=test_data[!, :PassengerId], Survived=predictions)

CSV.write("output/rand_forest_submission1.csv", submission)