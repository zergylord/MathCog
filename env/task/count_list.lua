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
--if wrong, teach
function task.step(state,act,say,prev_said)
    timer:reset()
    local term = false
    local r = torch.Tensor{{0}}
    --correct say indexing, then check its +1
    --or the beginning of counting (last was command)
    local next_numer
    if (prev_said > max_count) then
        next_numer = 1
    else
        next_numer = prev_said + 1
    end
    if command_steps == 0 then
        --said command, not numeral
        if (say-1) > max_count then
            r = torch.Tensor{{-1}}
            term = true
        elseif (say-1) == next_numer then 
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
    local teach
    if r[1][1] == -1 then
        teach = {torch.Tensor{{1}},torch.Tensor{{next_numer+1}}} 
    end
    return r,term,teach
end
return task
