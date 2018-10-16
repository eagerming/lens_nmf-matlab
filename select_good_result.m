function [TopK_result, TopKindex] = select_good_result( final_result, topN, mcnt)
    fields = fieldnames(final_result{1});
    numOf_K = length(final_result{1});
    
    if length(mcnt) == 1
        mcnt = 1:mcnt;
    end
    
    
    
    y_mean = zeros(length(mcnt),1);
    for idx = 1:length(mcnt)
        for i = 1 : length(fields)
            for k = 1:numOf_K
                y_mean(idx) = y_mean(idx) + final_result{idx}(k).(fields{i})...
                    / numOf_K / length(fields);
            end
        end
    end
   
    [~, TopKindex] = maxk(y_mean, topN);
    TopK_result = final_result(TopKindex); 
end