local task = {}
local max_count
local command_steps
local timer = torch.Timer()
--give ... is shorthand for give all you can
function task.init(max_a,state)
    max_count = max_a*2
    command = {torch.Tensor{{max_count+1+1+3}}}--give
    command_steps = #command
    return state,command
end
--positive reward for each correct next numeral
--negative reward for incorrect and start over
--TODO:add teaching here
function task.step(state,act,say,prev_said)
    timer:reset()
    local term = false
    local r = torch.Tensor{{0}}
    --correct say indexing, then check its +1
    --or the beginning of counting (last was command)
    if command_steps == 0 then
        --said command, not numeral
        if (say-1) > max_count then
            r = torch.Tensor{{-1}}
            term = true
        elseif ((prev_said > max_count) and (say-1) == 1) or ((say-1) == (prev_said+1)) then 
            r = torch.Tensor{{1}}
            --end of count list
            if (say-1) == max_count then
                term = true
            end
        else
            r = torch.Tensor{{-1}}
            term = true
        end
    else
        command_steps = command_steps -1
    end
    --print(timer:time().real)
    return r,term
end
return task
