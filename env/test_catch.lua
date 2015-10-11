catch = require 'catch'
ProFi = require '../ProFi'
ProFi:start()
catch.init()
while not term do
    state,r,term,image = catch.step(torch.Tensor{{2}})
end
ProFi:stop()
ProFi:writeReport('foo.txt')
