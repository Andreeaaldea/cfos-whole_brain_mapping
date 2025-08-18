function desc = getDescendants(G, node)
% Return all descendants of 'node' in digraph G (via successors)
    visited = containers.Map('KeyType','char','ValueType','logical');
    stack = {char(node)};
    desc = strings(0,1);
    while ~isempty(stack)
        u = stack{end}; stack(end) = [];
        sucs = successors(G, u);
        for i = 1:numel(sucs)
            v = char(sucs{i});
            if ~isKey(visited, v)
                visited(v) = true;
                desc(end+1,1) = string(v); %#ok<AGROW>
                stack{end+1} = v; %#ok<AGROW>
            end
        end
    end
end

function out = getOrEmpty(mapObj, keyStr)
% Safe map lookup that returns "" if key is missing
    key = char(string(keyStr));
    if isKey(mapObj, key), out = string(mapObj(key));
    else, out = "";
    end
end