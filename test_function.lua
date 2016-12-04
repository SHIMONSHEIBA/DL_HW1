local mnist = require 'mnist';
require 'nn'
require 'cunn'
require 'optim'

model = torch.load('ourModel.model')

testData = mnist.testdataset().data:float();
testLabels = mnist.testdataset().label:add(1);

function TestModel()
	
	local confusion = optim.ConfusionMatrix(torch.range(0,9):totable())
	local lossAcc = 0
	local numBatches = 0
	local batchSize = 16
	
	criterion = nn.CrossEntropyCriterion():cuda()
	
	model:evaluate()
	
    for i = 1, testData:size(1) - batchSize, batchSize do
        numBatches = numBatches + 1
        local x = testData:narrow(1, i, batchSize):cuda()
        local yt = testLabels:narrow(1, i, batchSize):cuda()
        local y = model:forward(x)
        local err = criterion:forward(y, yt)
        lossAcc = lossAcc + err
        confusion:batchAdd(y,yt)
        
    end
    
    confusion:updateValids()
    local avgError = 1 - confusion.totalValid
    return avgError
end


testError = TestModel()
print(testError)
