function saveSum(fidpath, mname, RMSE_min, RMSE_sloma)
    
    pos = strfind(fidpath,'/');
    if isempty(pos)
        pos = strfind(fidpath,'\');
    end
    name = extractAfter(fidpath,pos);
%     pos = strfind(name,'.');
%     name = extractBefore(name,pos);
    name = ['sum_' char(name) '.txt'];
    
    saveName = fullfile( "log" , name);


    fid = fopen(saveName,'w');
    
    %% 
    j = 1;
    for i = 1:length(mname)
        fprintf(fid, "%f\t%s\n", RMSE_min(i), mname{i});
        if contains(mname{i},'BoostCF')
            fprintf(fid, "%f\t%s\n", full(RMSE_sloma(j)), "Sloma");
            j = j+1;
        end
    end
    fclose(fid);
end

