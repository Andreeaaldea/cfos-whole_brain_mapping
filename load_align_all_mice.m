%% Build per-animal mean cells/mm^3 and merge by region
% Folder layout:
% baseDir / <group_name> / <mouse_number> / cellfinder/brainreg_trained_1/analysis/summary.csv
% Keeps only left_cells_per_mm3 & right_cells_per_mm3, averages them per region per animal.

clear; clc;
%%
% --- Config ---
baseDir    = 'Y:\public\projects\AnAl_20240405_Neuromod_PE\brainsaw\PE_mapping';
groupNames = {'WT_PE_bl2','WT_CT','Shank3_PE','Shank3_CT'};  

% Expected column names
regionCol = 'structure_name';
leftCol   = 'left_cells_per_mm3';
rightCol  = 'right_cells_per_mm3';
%%
% --- Collect per-mouse tables (LONG with hemisphere) ---
rows = {};  % cell array of per-mouse long tables

for g = 1:numel(groupNames)
    gname = groupNames{g};
    gdir  = fullfile(baseDir, gname);
    if ~isfolder(gdir)
        warning('Group folder not found: %s (skipping)', gdir);
        continue;
    end

    mice = dir(gdir);
    mice = mice([mice.isdir] & ~ismember({mice.name},{'.','..'}));

    for m = 1:numel(mice)
        mname   = mice(m).name;
        csvPath = fullfile(gdir, mname, 'cellfinder', 'brainreg_trained_1', 'analysis', 'summary.csv');

        if ~isfile(csvPath)
            fprintf('Missing summary for %s/%s — skipping.\n', gname, mname);
            continue;
        end

        T = readtable(csvPath, 'TextType','string');

        % sanity
        needed = {regionCol, leftCol, rightCol};
        if ~all(ismember(needed, T.Properties.VariableNames))
            warning('Missing required columns in %s. Found: %s', csvPath, strjoin(T.Properties.VariableNames, ', '));
            continue;
        end

        % keep only the three columns
        T = T(:, needed);
        T.(regionCol) = strtrim(T.(regionCol));

        % if duplicate regions exist, average within hemisphere BEFORE stacking
        if numel(T.(regionCol)) ~= numel(unique(T.(regionCol)))
            T = groupsummary(T, regionCol, 'mean', {leftCol, rightCol});
            % rename back
            T.GroupCount = [];
            T.Properties.VariableNames{strcmp(T.Properties.VariableNames, ['mean_' leftCol])}  = leftCol;
            T.Properties.VariableNames{strcmp(T.Properties.VariableNames, ['mean_' rightCol])} = rightCol;
        end

        % reshape to LONG with hemisphere factor
        TL = table(T.(regionCol), repmat("L",height(T),1), T.(leftCol), ...
                   'VariableNames', {regionCol,'hemisphere','cells_per_mm3'});
        TR = table(T.(regionCol), repmat("R",height(T),1), T.(rightCol), ...
                   'VariableNames', {regionCol,'hemisphere','cells_per_mm3'});

        U = [TL; TR];
        U.group = repmat(string(gname), height(U), 1);
        U.mouse = repmat(string(mname), height(U), 1);

        % reorder: group, mouse, region, hemisphere, value
        U = movevars(U, {'group','mouse'}, 'Before', regionCol);

        rows{end+1} = U; %#ok<AGROW>
    end
end

if isempty(rows)
    error('No CSVs found. Check baseDir and group folders.');
end

allLong = vertcat(rows{:});

% Save LONG
saveDir = 'Y:\public\projects\AnAl_20240405_Neuromod_PE\PE_mapping\processed_data'
outLong = fullfile(saveDir, 'cells_per_mm3_LEFT_RIGHT_LONG.csv');
writetable(allLong, outLong);
fprintf('Saved long table: %s\n', outLong);

%% Build analysis tables from LONG format
% Inputs
longCsv = 'Y:\public\projects\AnAl_20240405_Neuromod_PE\PE_mapping\processed_data\cells_per_mm3_LEFT_RIGHT_LONG';            
% Outputs
outWideMouse   = 'WIDE_by_mouse_from_LONG.csv';
outGroupMeans  = 'GROUP_MEANS_by_region_hemi_from_LONG.csv';
outGroupCounts = 'GROUP_COUNTS_by_region_hemi_from_LONG.csv';

% --- Load & sanity ---
T = readtable(longCsv, 'TextType','string');

% Expected columns in LONG table:
req = {'group','mouse','hemisphere','structure_name','cells_per_mm3'};
missing = setdiff(req, T.Properties.VariableNames);
if ~isempty(missing)
    error('LONG table is missing required columns: %s', strjoin(missing, ', '));
end

% Normalize hemisphere labels and region names
T.hemisphere     = upper(strtrim(T.hemisphere));
T.hemisphere( T.hemisphere=="LEFT" )  = "L";
T.hemisphere( T.hemisphere=="RIGHT" ) = "R";
T.structure_name = strtrim(T.structure_name);

% Coerce numeric values (just in case)
T.cells_per_mm3 = double(T.cells_per_mm3);

% --- De-duplicate within (group, mouse, hemisphere, structure_name): average ---
G = findgroups(T.group, T.mouse, T.hemisphere, T.structure_name);
U = unique(T(:, {'group','mouse','hemisphere','structure_name'}));
U.cells_per_mm3 = splitapply(@(x) mean(x, 'omitnan'), T.cells_per_mm3, G);

% --- WIDE by mouse: columns = <group>_<mouse>_<L/R> ---
U.colname = matlab.lang.makeValidName(strcat(U.group, "_", U.mouse, "_", U.hemisphere));
W = unstack(U(:, {'structure_name','colname','cells_per_mm3'}), ...
            'cells_per_mm3', 'colname');
W = sortrows(W, 'structure_name');

% Save
writetable(W, outWideMouse);
fprintf('Saved per-mouse WIDE: %s\n', outWideMouse);

% --- GROUP MEANS by region × hemisphere: columns = <group>_<L/R> (means) ---
Smean = groupsummary(U, {'structure_name','group','hemisphere'}, 'mean', 'cells_per_mm3');
Smean.ghemi = strcat(Smean.group, "_", Smean.hemisphere);
Means = unstack(Smean(:, {'structure_name','ghemi','mean_cells_per_mm3'}), ...
                'mean_cells_per_mm3', 'ghemi');
Means = sortrows(Means, 'structure_name');
writetable(Means, outGroupMeans);
fprintf('Saved GROUP MEANS: %s\n', outGroupMeans);


%% Join WIDE tables with brainmapper metadata
metaCsv     = 'brainmapper_base_architecture.csv';
inWideMouse = 'WIDE_by_mouse_from_LONG.csv';
inMeans     = 'GROUP_MEANS_by_region_hemi_from_LONG.csv';

outWideMouseMeta = 'WIDE_by_mouse_with_meta.csv';
outMeansMeta     = 'GROUP_MEANS_with_meta.csv';

A = readtable(metaCsv, 'TextType','string');

% Choose the name column present in metadata
if ismember('structure_name', A.Properties.VariableNames)
    metaNameCol = 'structure_name';
elseif ismember('name', A.Properties.VariableNames)
    metaNameCol = 'name';
else
    error('Metadata must contain either "structure_name" or "name".');
end

% Tidy subset of metadata (unique rows)
metaKeep = intersect({'region_id','acronym','name','structure_name','parent_structure_id','depth','structure_id_path'}, ...
                     A.Properties.VariableNames);
Meta = unique(A(:, metaKeep), 'rows');

% Attach meta to each table and save
W  = readtable(inWideMouse, 'TextType','string');
WM = attachMeta(W, Meta, metaNameCol);
% Remove rows with NaN in 'depth' column
WM = WM(~isnan(WM.depth), :);
writetable(WM, outWideMouseMeta);

M  = readtable(inMeans, 'TextType','string');
MM = attachMeta(M, Meta, metaNameCol);
WM = WM(~isnan(WM.depth), :);
writetable(MM, outMeansMeta);


fprintf('Saved:\n  %s\n  %s\n  %s\n', outWideMouseMeta, outMeansMeta);
