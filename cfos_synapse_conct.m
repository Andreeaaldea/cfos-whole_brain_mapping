% This is a continuation of the
% cellfinder_effectsize_hist_cellden_papexcir.m file assunimg that all the
% values have been loaded
%this code claculates the effect size across all mice
%%
% data = 10 animals (PE WT bl2 Cl WT) x all (672) regions

data = [dCellsExperiment1, dCellsExperiment2,dCellsExperiment3,dCellsExperiment4,dCellsExperiment5,...
    dCellsControl1, dCellsControl2,dCellsControl3,dCellsControl4,dCellsControl5]
data= data';

% data: rows = animals, columns = regions

% 1 Normalize each region across all animals
regionMean = mean(data, 1);       % 1 x 672 vector of means per region
regionStd  = std(data, 0, 1);       % 1 x 672 vector of standard deviations per region
zScores = (data - regionMean) ./ regionStd;

%2 Separate groups based on rows
experimental = zScores(1:5, :);
control = zScores(6:10, :);

% 3 Calculate group means for each region
meanControl = mean(control, 1);        % 1 x 672 vector
meanExperimental = mean(experimental, 1);

% 4 Compute pooled standard deviation for each region
nControl = size(control, 1);
nExperimental = size(experimental, 1);
stdControl = std(control, 0, 1);
stdExperimental = std(experimental, 0, 1);
pooledStd = sqrt(((nExperimental - 1) .* (stdExperimental.^2) + (nControl - 1) .* (stdControl.^2)) / ( nExperimental + nControl - 2));

% 5 Calculate Cohen's d for each region
cohens_d = (meanExperimental - meanControl) ./ pooledStd;


Index = (1:numel(cohens_d))';
Region_Name = actrontable.structure_name;
meanzscoredenPE = meanExperimental';
meanzscoredenControl = meanControl';
eachregcohens_d = cohens_d';

dbrainDatanew = table(Index, Region_Name, meanzscoredenPE, meanzscoredenControl, eachregcohens_d);
%%
%load the synapse data

locsyn1 = uigetdir([], 'Select folder for LCsyn1');
locsyn2 = uigetdir([], 'Select folder for LCsyn2');
locsyn3= uigetdir([], 'Select folder for LCsyn3');
dataFilePathlcsyn1 = fullfile(locsyn1, 'region_quantification_withnames_newmask_gr_HR.csv');
dataFilePathlcsyn2 = fullfile(locsyn2, 'region_quantification_withnames.csv');
dataFilePathlcsyn3 = fullfile(locsyn3, 'region_quantification_withnames.csv');
dataTsyn11 = readtable(dataFilePathlcsyn1);
dataTsyn12 = readtable(dataFilePathlcsyn2);
dataTsyn13 = readtable(dataFilePathlcsyn3);

%%
% mean and z score syn data
syn_dataT1 = innerjoin(dataTsyn11, dataTsyn12,'Keys','Region_Name');
syn_dataT = innerjoin(syn_dataT1, dataTsyn13,'Keys','Region_Name');
syn_dataT = removevars(syn_dataT, ["id_dataTsyn11","Pixel_Count_dataTsyn11","id_dataTsyn12","Pixel_Count_dataTsyn12","id","Pixel_Count"]);
LC_Syn_mean = [syn_dataT.Mean_Intensity,syn_dataT.Mean_Intensity_dataTsyn11, syn_dataT.Mean_Intensity_dataTsyn12] ;

%%
%merge tables
% zscored cfos data 
syncfosT = innerjoin(syn_dataT, dbrainDatanew, 'Keys', 'Region_Name');%, 'MergeKeys', true);

%effect size cfos data
syncfosT_chd  = innerjoin(syn_dataT, dstandarddata, 'Keys', 'Region_Name');%

%sort form highest effect size
Tchd_zsc_acron = sortrows(syncfosT_chd, 'dNormcohen_d', 'descend');

%%
%All brain regions (no white matter)

%Bin the lables
Reg_Name = Tchd_zsc_acron.Region_Name;
Effect_sise_cells = Tchd_zsc_acron.dNormcohen_d;

%Acronym = Tchd_zsc_acron.acronym;
%Index = Tchd_zsc_acron.Idx;

%% normalize the lc data
LC_Syn_mean = [Tchd_zsc_acron.Mean_Intensity,Tchd_zsc_acron.Mean_Intensity_dataTsyn11, Tchd_zsc_acron.Mean_Intensity_dataTsyn12] ;
%z-scored
z_data = (LC_Syn_mean - mean(LC_Syn_mean, 1)) ./ std(LC_Syn_mean);
%min-max
min_vals = min(LC_Syn_mean);
max_vals = max(LC_Syn_mean);
range_vals = max_vals - min_vals;
minmax_data = (LC_Syn_mean - min_vals) ./ range_vals;
% divide by mean
mean_norm_data  = LC_Syn_mean ./ mean(LC_Syn_mean); % or data ./ max(data)

z_vec = z_data(:);
minmax_vec = minmax_data(:);
mean_vec = mean_norm_data(:);

figure;
histogram(z_vec, 'Normalization', 'pdf', 'DisplayStyle', 'stairs'); hold on;
histogram(minmax_vec, 'Normalization', 'pdf', 'DisplayStyle', 'stairs');
histogram(mean_vec, 'Normalization', 'pdf', 'DisplayStyle', 'stairs');
legend('Z-score', 'Min-Max', 'Mean-normalized');
title('Comparison of Normalization Methods');
xlabel('Normalized Value');
ylabel('Probability Density');

% keep z-scored

Tchd_zsc = Tchd_zsc_acron;
Tchd_zsc.LC_Syn_zsc = mean(z_data,2);
Tchd_zsc = removevars(Tchd_zsc, ["Mean_Intensity_dataTsyn11","Total_Intensity_dataTsyn11","Mean_Intensity_dataTsyn12","Total_Intensity_dataTsyn12","Mean_Intensity","Total_Intensity"]);


selectarea = Tchd_zsc.Effect_sise_cells>= 0.8 & Tchd_zsc.LC_Syn_zsc>= 0.8;
highreg= find(selectarea);
Thighreg = Tchd_zsc(highreg,:);

selectarea = Tchd_zsc.Effect_sise_cells<0 & Tchd_zsc.LC_Syn_zsc>=1;
lowreg= find(selectarea);
Tlowreg = Tchd_zsc(lowreg,:);


writetable(Tchd_zsc,'cfos_lc_syn.csv')


figure;
%scatterhist(Effect_sise_cells(highreg),LC_Syn_zsc(highreg))
sc = scatter(Effect_sise_cells,LC_Syn_zsc)
distr = fit(Effect_sise_cells,LC_Syn_zsc,'poly2')
hold on 
plot (distr)
xlabel('Effect size cfos-cells');ylabel('LC Synapses z-scored');
title('Cfos positive cells and LC synapses.', 'FontSize', 14, 'FontWeight', 'bold');
colororder("meadow")
ylim([-3 3]); xlim([-3 3])
legend off


figure;
%scatterhist(Effect_sise_cells(highreg),LC_Syn_zsc(highreg))
sc = scatter(Effect_sise_cells(highreg),LC_Syn_zsc(highreg))
distr = fit(Effect_sise_cells(highreg),LC_Syn_zsc(highreg),'poly2')
hold on 
plot (distr)
xlabel('Effect size cfos-cells');ylabel('LC Synapses z-scored');
title('High density of cfos positive cells and LC synapses.', 'FontSize', 14, 'FontWeight', 'bold');
colororder("meadow")
legend off

%%
x = Effect_sise_cells;
y = LC_Syn_zsc;
labels = Acronym;  

figure;
sc = scatter(x, y, 60, 'filled'); 
hold on

% fit and plot the polynomial trend
distr = fit(x, y, 'poly2');
hLine = plot(distr);
hLine.Color = [0.2 0.2 0.2];         % dark grey line
hLine.LineWidth = 1.5;

% annotate each point
dx = 0.005 * range(x);   % horizontal offset = 2% of x-range
dy = 0.005 * range(y);   % vertical offset = 2% of y-range
for i = 1:numel(x)
    text(x(i)+dx, y(i)+dy, labels{i}, ...
         'FontSize', 12, ...
         'Interpreter', 'none', ...
         'Rotation', 45, ...                           
         'HorizontalAlignment','left', ...             
         'VerticalAlignment','bottom', ...             
         'Color', [0.01 0.01 0.01] );
end

xlabel('Effect size cfos-cells');
ylabel('LC Synapses z-scored');
title('High density of cfos positive cells and LC synapses',...
      'FontSize', 14, 'FontWeight', 'bold');

colororder("meadow")
legend off
axis tight
%ylim([0.75 2.25]);xlim([0.75 2.25])
box on


%%

figure;
scatterhist(Effect_sise_cells,datasyn)
ylabel('mean_syn');xlabel('effect_size') 
%ylim([-5 5])
% bin the chd values in bins of 0.5 and plot box plots

boxplot
% Create a bar plot with the assigned colors
figure;
b = bar(categorical(Reg_Name(1:40,:)), Effect_sise_cells(1:40,:), 'FaceColor', 'flat', 'EdgeColor', 'none');
hold on
%b = bar(categorical(regplot), dataplot, 'FaceColor', 'flat', 'EdgeColor', 'none');
b = bar(LC_Syn_zsc(1:40,:));%, 'FaceColor', 'flat', 'EdgeColor', 'none');
%yline(meanefsize, 'r--', 'LineWidth', 2); % 'r--' specifies a red dashed line, 'LineWidth' sets its width
% Apply the colors to the bars
b.CData = barColors;

% Add grid lines
grid on;
grid minor;

% Enhance the x-axis and y-axis labels
xlabel('Brain Region', 'FontSize', 14, 'FontWeight', 'bold');
ylabel('Cohens_d', 'FontSize', 14, 'FontWeight', 'bold');

% Enhance the title
title('Effect size.Active Cells (cell density). Normalised', 'FontSize', 16, 'FontWeight', 'bold');

% Rotate the x-axis labels
set(gca, 'XTickLabelRotation', 45, 'FontSize', 12);
