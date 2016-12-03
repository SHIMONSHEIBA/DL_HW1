local mnist = require 'mnist';
local mnist = require 'mnist';

local trainData = mnist.traindataset().data:float();
local trainLabels = mnist.traindataset().label:add(1);
testData = mnist.testdataset().data:float();
testLabels = mnist.testdataset().label:add(1);

--We'll start by normalizing our data
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

model:add(nn.Linear(layerSize[#layerSize], outputSize))
model:add(nn.LogSoftMax())   -- f_i(x) = exp(x_i - shift) / sum_j exp(x_j - shift)


model:cuda() --ship to gpu
print(tostring(model))

local w, dE_dw = model:getParameters()
print('Number of parameters:', w:nElement()) --over-specified model


---- ### Classification criterion

criterion = nn.ClassNLLCriterion():cuda()
--criterion = nn.MSECriterion():cuda()

---	 ### predefined constants

require 'optim'
batchSize = 16

optimState = {
    learningRate = 0.1   
}

--- ### Main evaluation + training function


function forwardNet(data, labels, train, e)
	timer = torch.Timer()

    --another helpful function of optim is ConfusionMatrix
    local confusion = optim.ConfusionMatrix(torch.range(0,9):totable())
    local lossAcc = 0
    local numBatches = 0
    if train then
        --set network into training mode
        model:training()
    end
    for i = 1, data:size(1) - batchSize, batchSize do
        numBatches = numBatches + 1
	--print('Number of batches:', numBatches)
        local x = data:narrow(1, i, batchSize):cuda()
	--print('Is it here?1')
        local yt = labels:narrow(1, i, batchSize):cuda()
	--print('yt type is:',type(yt))
	--print('Is it here?2')
        local y = model:forward(x)
	--print('y type is:',type(y))
	--print('Is it here?3')
        local err = criterion:forward(y, yt)
	--if numBatches == 1 then
		--print('error is:' , err)
	--end
	--print('Is it here?4')
	lossAcc = lossAcc + err
	--if numBatches == 1 then
		--print('lossAcc is:',lossAcc)
	--end
	--print('Is it here?5')
	--print(y:dim(),yt:dim(),'y type is:', y.type,'yt type is:',yt.type)
        confusion:batchAdd(y,yt)
        --print('Is it here?6')       
        if train then
            function feval()
                model:zeroGradParameters() --zero grads
                local dE_dy = criterion:backward(y,yt)
                model:backward(x, dE_dy) -- backpropagation
            
                return err, dE_dw
            end
        --print('Is it here?7')
            optim.sgd(feval, w, optimState)
        end
    end
    
    confusion:updateValids()
	--print('Is it here?8')
    local avgLoss = lossAcc / numBatches
    local avgError = 1 - confusion.totalValid
	print('epoc: ' ..e, timer:time().real .. ' seconds')

    return avgLoss, avgError, tostring(confusion)
end



--- ### Train the network on training set, evaluate on separate set


epochs = 20

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
    
    if e % 5 == 0 then
        print('Epoch ' .. e .. ':')
        print('Training error: ' .. trainError[e], 'Training Loss: ' .. trainLoss[e])
        print('Test error: ' .. testError[e], 'Test Loss: ' .. testLoss[e])
        print(confusion)
    end
end



---		### Introduce momentum, L2 regularization
--reset net weights
model:apply(function(l) l:reset() end)

optimState = {
    learningRate = 0.1,
    momentum = 0.9,
    weightDecay = 1e-3   
}
for e = 1, epochs do
    trainData, trainLabels = shuffle(trainData, trainLabels) --shuffle training data
    trainLoss[e], trainError[e] = forwardNet(trainData, trainLabels, true, e)
    testLoss[e], testError[e], confusion = forwardNet(testData, testLabels, false, e)
end

print('Training error: ' .. trainError[epochs], 'Training Loss: ' .. trainLoss[epochs])
print('Test error: ' .. testError[epochs], 'Test Loss: ' .. testLoss[epochs])




--- ### Insert a Dropout layer
--[[
model:insert(nn.Dropout(0.9):cuda(), 8)







-- ********************* Plots *********************

require 'gnuplot'
local range = torch.range(1, epochs)
gnuplot.pngfigure('test.png')
gnuplot.plot({'trainLoss',trainLoss},{'testLoss',testLoss})
gnuplot.xlabel('epochs')
gnuplot.ylabel('Loss')
gnuplot.plotflush()
]]














