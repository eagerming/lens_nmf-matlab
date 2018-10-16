function [fid, fidpath] = create_new_log(log_name)
    if ~exist('log_name', 'var')
        log_name = 'Mylog';
    end
    
    i = 1;
    fid = [];
    
    while true
        name = strcat(log_name, '_', num2str(i), '.log');
        fidpath = fullfile( "log" , name);
        isfile = dir(fidpath);
        
        if isempty(isfile) % the name can be created;
            fid = fopen(fidpath, 'a');
            break;
        end
        
        i = i + 1;
    end
end