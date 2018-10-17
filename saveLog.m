function saveLog(fid, final_result, K_list, mname)
    
    fields = fieldnames(final_result{1});
    numOf_K = length(final_result{1});
    mcnt = length(final_result);
    
    
    
    if length(mcnt) == 1
        mcnt = 1:mcnt;
    end
    
    
    y_mean = zeros(length(mcnt), length(fields));
    for idx = 1:length(mcnt)
        for i = 1 : length(fields)
            for k = 1:numOf_K
                y_mean(idx, i) = y_mean(idx, i) + final_result{idx}(k).(fields{i}) ...
                    / numOf_K;
            end
        end
    end
    method_score = mean(y_mean,2);
    [~, index] = sort(method_score, 'descend');
    
    %% 
    fprintf(fid, "\n===================== Average measures over K======================\n");
    for i = 1 : length(fields)
        fprintf(fid, strcat(fields{i}, "\t"));
    end
    
    fprintf(fid, "\n");
    for i = 1 : length(mcnt)
        fprintf(fid, mname{i});
        fprintf(fid, "\t");
        for j = 1 :  length(fields)
            fprintf(fid, num2str(y_mean(index(i),j)));
            fprintf(fid, "\t");
        end
        fprintf(fid, "\n");
    end
    
    
    %%
    [maxK, maxId] = max(K_list);
    
    fprintf(fid, "\n");
    fprintf(fid, "\n================ Largest measures over max_K=%d ==================\n", maxK);
    for i = 1 : length(fields)
        fprintf(fid, strcat(fields{i}, "\t"));
    end
    
    fprintf(fid, "\n");
    for i = 1 : length(mcnt)
        fprintf(fid, mname{i});
        fprintf(fid, "\t");
        for j = 1 : length(fields)
            fprintf(fid, num2str(final_result{index(i)}(maxId).(fields{j})));
            fprintf(fid, "\t");
        end
        fprintf(fid, "\n");
    end
    
end

