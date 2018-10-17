function saveName = saveResult(fidpath)
    pos = strfind(fidpath,'/');
    name = extractAfter(fidpath,pos);
    pos = strfind(name,'.');
    name = extractBefore(name,pos);
    
    saveName = fullfile( "result" , name);
    
end