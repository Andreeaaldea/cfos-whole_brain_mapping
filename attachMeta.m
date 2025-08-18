function Tm = attachMeta(T, Meta, metaNameCol)
%ATTACHMETA Attach atlas metadata to a region-by-* table.
%   Tm = ATTACHMETA(T, Meta, metaNameCol)
%   - Prefers joining by region_id if present in both tables.
%   - Otherwise joins by normalized structure_name (spaces collapsed, lowercased).
%
%   Inputs:
%     T            table with at least structure_name (or region_id)
%     Meta         metadata table (e.g., brainmapper_base_architecture.csv)
%     metaNameCol  which name column in Meta to use ('structure_name' or 'name')

    % Prefer region_id join if available in both
    if ismember('region_id', T.Properties.VariableNames) && ...
       ismember('region_id', Meta.Properties.VariableNames)

        Tm = outerjoin(Meta, T, 'Keys', 'region_id', 'MergeKeys', true);

    else
        if ~ismember('structure_name', T.Properties.VariableNames)
            error('Input table must contain "structure_name" to join by name.');
        end
        if ~ismember(metaNameCol, Meta.Properties.VariableNames)
            error('Meta table is missing the column "%s".', metaNameCol);
        end

        % Build normalized join keys (lowercase, single spaces)
        T.join_key    = lower(regexprep(strtrim(string(T.structure_name)), '\s+', ' '));
        Meta.join_key = lower(regexprep(strtrim(string(Meta.(metaNameCol))), '\s+', ' '));

        Tm = outerjoin(Meta, T, 'Keys', 'join_key', 'MergeKeys', true);

        % Drop helper key if present
        if ismember('join_key', Tm.Properties.VariableNames)
            Tm.join_key = [];
        end
    end

    % Put meta columns up front for readability
    lead   = intersect({'region_id','acronym','name','structure_name'}, Tm.Properties.VariableNames, 'stable');
    others = setdiff(Tm.Properties.VariableNames, lead, 'stable');
    Tm = Tm(:, [lead, others]);
end
