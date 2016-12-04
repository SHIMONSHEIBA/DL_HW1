local mnist = require 'mnist';
local mnist = require 'mnist';

local trainData = mnist.traindataset().data:float();
local trainLabels = mnist.traindataset().label:add(1);
testData = mnist.testdataset().data:float();
testLabels = mnist.testdataset().label:add(1);

--normalizing our data
local mean = trainData:mean()
local std = trainData:std()
trainData:add(-mean):div(std); 
testData:add(-mean):div(std);


----- ### Shuffling data

function shuffle(data, labels) --shuffle data function
    local randomIndexes = torch.randperm(data:size(1)):long() 
    return data:index(1,randomIndexes), labels:index(1,randomIndexes)
end

------   ### Define model and criterion

require 'nn'
require 'cunn'

local inputSize = 28*28
local outputSize = 10
local layerSize = {inputSize,64,64,64,64}

model = nn.Sequential()
model:add(nn.View(28 * 28)) --reshapes the image into a vector without copy
for i=1, #layerSize-1 do
    model:add(nn.Linear(layerSize[i], layerSize[i+1]))
    model:add(nn.LeakyReLU())
end

--add dropout function before the last linear layer
model:add(nn.Dropout(0.2):cuda(), 8)
model:add(nn.Linear(layerSize[#layerSize], outputSize))
model:add(nn.LogSoftMax())   -- f_i(x) = exp(x_i - shift) / sum_j exp(x_j - shift)


model:cuda() --ship to gpu
print(tostring(model))

local w, dE_dw = model:getParameters()
print('Number of parameters:', w:nElement()) --over-specified model


---- ### Classification criterion

criterion = nn.CrossEntropyCriterion():cuda()

---	 ### predefined constants

require 'optim'
batchSize = 16

optimState = {
	learningRate = 0.001
}

--- ### Main evaluation + training function


function forwardNet(data, labels, train, e)
	timer = torch.Timer()

    --Declare ConfusionMatrix
    local confusion = optim.ConfusionMatrix(torch.range(0,9):totable())
    local lossAcc = 0
    local numBatches = 0
    if train then
        --set network into training mode
        model:training()
    else
        model:evaluate() -- turn of drop-out
    end
	--forward part
    for i = 1, data:size(1) - batchSize, batchSize do
        numBatches = numBatches + 1
        local x = data:narrow(1, i, batchSize):cuda()
        local yt = labels:narrow(1, i, batchSize):cuda()
        local y = model:forward(x)
        local err = criterion:forward(y, yt)
	lossAcc = lossAcc + err
        confusion:batchAdd(y,yt)      
        if train then
		--backward part
            function feval()
                model:zeroGradParameters() --zero grads
                local dE_dy = criterion:backward(y,yt)
                model:backward(x, dE_dy) -- backpropagation
            
                return err, dE_dw
            end
		--optimization function to use
            optim.sgd(feval, w, optimState)
        end
    end
    
    confusion:updateValids()
    local avgLoss = lossAcc / numBatches
    local avgError = 1 - confusion.totalValid
	print('epoc: ' ..e, timer:time().real .. ' seconds')

    return avgLoss, avgError, tostring(confusion)
end



--- ### Train the network on training set, evaluate on separate set


epochs = 75

trainLoss = torch.Tensor(epochs)
testLoss = torch.Tensor(epochs)
trainError = torch.Tensor(epochs)
testError = torch.Tensor(epochs)

--reset net weights
model:apply(function(l) l:reset() end)

for e = 1, epochs do
    trainData, trainLabels = shuffle(trainData, trainLabels) --shuffle training data
    trainLoss[e], trainError[e] = forwardNet(trainData, trainLabels, true, e)
    testLoss[e], testError[e], confusion = forwardNet(testData, testLabels, false, e)
    
    if e > 55 then
        print('Epoch ' .. e .. ':')
        print('Training error: ' .. trainError[e], 'Training Loss: ' .. trainLoss[e])
        print('Test error: ' .. testError[e], 'Test Loss: ' .. testLoss[e])
    end
end

--save th model we trained
torch.save('ourModel.model', model)


-- ********************* Plots *********************

require 'gnuplot'
local range = torch.range(1, epochs)
gnuplot.pngfigure('loss.png')
gnuplot.plot({'trainLoss',trainLoss},{'testLoss',testLoss})
gnuplot.xlabel('epochs')
gnuplot.ylabel('Loss')
gnuplot.plotflush()

gnuplot.pngfigure('error.png')
gnuplot.plot({'trainError',trainError},{'testError',testError})
gnuplot.xlabel('epochs')
gnuplot.ylabel('Error')
gnuplot.plotflush()















