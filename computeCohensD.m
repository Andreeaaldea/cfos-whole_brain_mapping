%% 1) Define a helper for Cohen’s d
function d = computeCohensD(group1, group2)
    % group1, group2 are R×n1 and R×n2
    m1 = mean( group1, 2, 'omitnan' );
    m2 = mean( group2, 2, 'omitnan' );
    s1 = std(  group1, 0, 2, 'omitnan' );
    s2 = std(  group2, 0, 2, 'omitnan' );
    n1 = sum(~isnan(group1), 2);
    n2 = sum(~isnan(group2), 2);
    sp = sqrt(((n1-1).*s1.^2 + (n2-1).*s2.^2) ./ (n1 + n2 - 2));
    d  = (m1 - m2) ./ sp;
end
