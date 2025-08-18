inXlsx = 'Y:\public\projects\AnAl_20240405_Neuromod_PE\PE_mapping\processed_data\dff_downsampled\mean_dff_per_region_by_mouse_brainmapper.xlsx'; % path to your file
T = readtable(inXlsx, 'Sheet', 1);

cols = {'region_id','acronym','name','structure_id_path','parent_structure_id','depth'};
missing = setdiff(cols, T.Properties.VariableNames);
if ~isempty(missing)
    error('Missing columns: %s', strjoin(missing, ', '));
end

Tsel = T(:, cols);
writetable(Tsel, 'brainmapper_base_architecture.csv');
