%% Collapse WIDE-by-mouse(+meta) table to TARGET_N regions and aggregate per-column
% Works with a WIDE CSV having meta columns and one numeric column per mouse/hemisphere.
clear; clc;

%% ===== CONFIG =====
structures_csv = 'structures.csv';
wide_with_meta_csv = 'WIDE_by_mouse_with_meta.csv';   
output_excel = 'final_collapsed_from_WIDE.xlsx';
output_mapcsv = 'collapsing_map.csv';                 % mapping of original region -> collapsed region

TARGET_N = 160;   % number of collapsed regions
root_id  = "997"; % Allen root for mouse

%% ===== Load Allen structures, build hierarchy, choose collapsed nodes =====
S = readtable(structures_csv, 'TextType','string');
needS = {'id','acronym','name','structure_id_path'};
missS = setdiff(needS, S.Properties.VariableNames);
if ~isempty(missS), error('structures.csv missing: %s', strjoin(missS, ', ')); end

% Normalize as strings
S.id = string(S.id);
S.structure_id_path = string(S.structure_id_path);

% Build graph edges from structure_id_path
src = strings(0,1); dst = strings(0,1);
for i = 1:height(S)
    parts = split(strtrim(S.structure_id_path(i)),"/");
    parts = parts(parts~="");
    for k = 1:numel(parts)-1
        src(end+1,1) = parts(k); %#ok<AGROW>
        dst(end+1,1) = parts(k+1); %#ok<AGROW>
    end
end
G = digraph(src, dst);

if ~any(G.Nodes.Name == root_id)
    error('Root id %s not found in graph.', root_id);
end

% Leaves that are reachable from root
od = outdegree(G);
leafNames = G.Nodes.Name(od==0);
leafNames = leafNames(arrayfun(@(x) isfinite(distances(G, root_id, x)), leafNames));

% Frequency of nodes across root->leaf shortest paths
counts = containers.Map('KeyType','char','ValueType','double');
for i = 1:numel(leafNames)
    try
        p = shortestpath(G, root_id, leafNames(i));
    catch
        p = string([]);
    end
    for node = p
        key = char(node);
        counts(key) = (isKey(counts,key))*counts(key) + 1;
    end
end

% Rank by frequency and greedily select non-overlapping ancestors
allKeys = string(keys(counts));
allVals = cell2mat(values(counts, cellstr(allKeys)));
[~,ix] = sort(allVals,'descend');
ranked = allKeys(ix);

selected = strings(0,1);
for n = ranked'
    desc = getDescendants(G, n);
    if ~any(ismember(desc, selected))
        selected(end+1,1) = n; %#ok<AGROW>
        if numel(selected) >= TARGET_N, break; end
    end
end
collapsed_ids = selected;

% ID -> name/acronym/path lookups
id2name    = containers.Map(cellstr(S.id), cellstr(S.name));
id2acr     = containers.Map(cellstr(S.id), cellstr(S.acronym));
id2path    = containers.Map(cellstr(S.id), cellstr(S.structure_id_path));

%% ===== Load WIDE+meta, identify columns =====
W = readtable(wide_with_meta_csv, 'TextType','string');

% Meta columns we care about (only those that actually exist are used)
metaCandidates = {'region_id','acronym','name','structure_id_path','depth','structure_name'};
metaCols = intersect(metaCandidates, W.Properties.VariableNames, 'stable');

% Ensure we can recover a path and an ID for each row
hasPath = ismember('structure_id_path', W.Properties.VariableNames);
hasID   = ismember('region_id', W.Properties.VariableNames);

if ~hasPath && ~hasID
    error('Input WIDE table must have either structure_id_path or region_id.');
end

% Identify numeric (value) columns = all non-meta numeric columns
isNum = varfun(@isnumeric, W, 'OutputFormat','uniform');
valueCols = W.Properties.VariableNames(isNum);
valueCols = setdiff(valueCols, intersect(valueCols, metaCols), 'stable');
if isempty(valueCols)
    error('No numeric value columns detected in WIDE table.');
end

% If path missing, reconstruct from region_id via structures.csv
if ~hasPath
    if ~hasID, error('Need region_id to reconstruct structure_id_path.'); end
    W.structure_id_path = strings(height(W),1);
    for i = 1:height(W)
        rid = W.region_id(i);
        if ~ismissing(rid)
            key = char(string(rid));
            if isKey(id2path, key)
                W.structure_id_path(i) = string(id2path(key));
            end
        end
    end
    hasPath = true;
    if ~hasPath
        error('Could not reconstruct structure_id_path from region_id for all rows.');
    end
end

%% ===== Map each row to a collapsed ancestor =====
W.collapsed_region_id = strings(height(W),1);

for i = 1:height(W)
    path = strtrim(W.structure_id_path(i));
    if path == "" || ismissing(path)
        continue;
    end
    parts = split(path,"/"); parts = parts(parts~="");
    % Walk from leaf up to root, pick first selected ancestor
    mapped = "";
    for k = numel(parts):-1:1
        if any(collapsed_ids == parts(k))
            mapped = parts(k); break;
        end
    end
    W.collapsed_region_id(i) = mapped;
end

% Drop rows that couldn't be mapped (rare; e.g., malformed paths)
W = W(W.collapsed_region_id ~= "", :);

%% ===== Aggregate per collapsed region across child rows (per value column) =====
% Mean & std across children per column
Smean = groupsummary(W, 'collapsed_region_id', 'mean', valueCols);
Sstd  = groupsummary(W, 'collapsed_region_id', 'std',  valueCols);
% Per-variable non-missing counts to compute SEM
Scount = varfun(@(x) sum(~isnan(x)), W, 'InputVariables', valueCols, ...
                'GroupingVariables', 'collapsed_region_id');
% Rename count cols to n_<var>
for c = 1:numel(valueCols)
    old = sprintf('Fun_%s', valueCols{c});
    if ismember(old, Scount.Properties.VariableNames)
        Scount.Properties.VariableNames{strcmp(Scount.Properties.VariableNames, old)} = sprintf('n_%s', valueCols{c});
    end
end

% Tidy mean/std tables (remove GroupCount, drop 'mean_'/'std_' prefixes)
if ismember('GroupCount', Smean.Properties.VariableNames), Smean.GroupCount = []; end
if ismember('GroupCount', Sstd.Properties.VariableNames),  Sstd.GroupCount  = []; end
for c = 1:numel(valueCols)
    mOld = sprintf('mean_%s', valueCols{c});
    sOld = sprintf('std_%s',  valueCols{c});
    if ismember(mOld, Smean.Properties.VariableNames)
        Smean.Properties.VariableNames{strcmp(Smean.Properties.VariableNames, mOld)} = valueCols{c};
    end
    if ismember(sOld, Sstd.Properties.VariableNames)
        Sstd.Properties.VariableNames{strcmp(Sstd.Properties.VariableNames, sOld)} = valueCols{c};
    end
end

% Compute SEM = std / sqrt(n) for each value column
Ssem = Sstd(:, {'collapsed_region_id'}); % start with key
for c = 1:numel(valueCols)
    sCol = valueCols{c};             % std col name (already renamed)
    nCol = sprintf('n_%s', sCol);    % count col name
    if ismember(sCol, Sstd.Properties.VariableNames) && ismember(nCol, Scount.Properties.VariableNames)
        nvec = Scount.(nCol);
        Ssem.(sCol) = Sstd.(sCol) ./ sqrt(max(nvec, 1));  % avoid div-by-zero
    end
end

% Attach collapsed meta to each table
CollapsedMeta = table( ...
    double(str2double(Smean.collapsed_region_id)), ...
    arrayfun(@(x) getOrEmpty(id2name, x), Smean.collapsed_region_id, 'uni', 0)', ...
    arrayfun(@(x) getOrEmpty(id2acr,  x), Smean.collapsed_region_id, 'uni', 0)', ...
    arrayfun(@(x) string(id2path(char(x))), Smean.collapsed_region_id), ...
    arrayfun(@(p) numel(split(p,"/"))-1, arrayfun(@(x) string(id2path(char(x))), Smean.collapsed_region_id)), ...
    'VariableNames', {'region_id','name','acronym','structure_id_path','depth'} ...
);

MeanOut = [CollapsedMeta Smean(:, setdiff(Smean.Properties.VariableNames, {'collapsed_region_id'}, 'stable'))];
StdOut  = [CollapsedMeta Sstd(:,  setdiff(Sstd.Properties.VariableNames,  {'collapsed_region_id'}, 'stable'))];
SemOut  = [CollapsedMeta Ssem(:,  setdiff(Ssem.Properties.VariableNames,  {'collapsed_region_id'}, 'stable'))];

% Sort by id for readability
[~,ord] = sort(MeanOut.region_id);
MeanOut = MeanOut(ord,:); StdOut = StdOut(ord,:); SemOut = SemOut(ord,:);

%% ===== Save outputs =====
writetable(MeanOut, output_excel, 'Sheet','mean', 'FileType','spreadsheet');
writetable(StdOut,  output_excel, 'Sheet','std',  'FileType','spreadsheet');
writetable(SemOut,  output_excel, 'Sheet','sem',  'FileType','spreadsheet');
fprintf('Collapsed matrices saved to: %s\n', output_excel);

% Save a simple mapping of original -> collapsed (for provenance)
Map = unique(W(:, {'region_id','structure_id_path','acronym','name','collapsed_region_id'}), 'rows');
Map.collapsed_name    = arrayfun(@(x) getOrEmpty(id2name, x), Map.collapsed_region_id, 'uni', 0)';
Map.collapsed_acronym = arrayfun(@(x) getOrEmpty(id2acr,  x), Map.collapsed_region_id, 'uni', 0)';
writetable(Map, output_mapcsv);
fprintf('Collapsing map saved to: %s\n', output_mapcsv);


