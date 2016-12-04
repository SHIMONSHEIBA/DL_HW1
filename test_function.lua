local mnist = require 'mnist';

print('1')
model = torch.load('ourModel.t7')
print(tostring(model))
print('2')
testData = mnist.testdataset().data:float();
testLabels = mnist.testdataset().label:add(1);
testError = TestModel()
print(testError)

function TestModel()
	print('3')
	local confusion = optim.ConfusionMatrix(torch.range(0,9):totable())
	local lossAcc = 0
	local numBatches = 0
	local batchSize = 16
	
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
