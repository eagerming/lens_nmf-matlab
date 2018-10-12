function  [R,S,map]= loadData(rating_filePath, trust_filePath, dataname)
%% read filmtrust data:

% rating_filePath = 'data/filmtrust/rating/ratings.txt';
% trust_filePath = 'data/filmtrust/trust/trust.txt';

rating_fileID = fopen(rating_filePath);
trust_fileID = fopen(trust_filePath);

rating = textscan(rating_fileID,'%f %f %f');
trust = textscan(trust_fileID,'%f %f %f');


rating = [rating{1,1}, rating{1,2}, rating{1,3}];
trust = [trust{1,1}, trust{1,2}, trust{1,3}];

%% Mapping

userID = trust(:,1:2);
userID = [userID(:); rating(:,1)];
map_from_colID_to_userID = unique(userID,'sorted');
map_from_userID_to_colID = containers.Map(map_from_colID_to_userID, 1:length(map_from_colID_to_userID));

itemID = rating(:,2);
map_from_rowID_to_itemID = unique(itemID,'sorted');
map_from_itemID_to_rowID = containers.Map(map_from_rowID_to_itemID, 1:length(map_from_rowID_to_itemID));

num_user = length(map_from_colID_to_userID);
num_item = length(map_from_rowID_to_itemID);

map.map_from_colID_to_userID = map_from_colID_to_userID;
map.map_from_userID_to_colID = map_from_userID_to_colID;
map.map_from_rowID_to_itemID = map_from_rowID_to_itemID;
map.map_from_itemID_to_rowID = map_from_itemID_to_rowID;

%% Transform the raw rating data to rating matrix R

R = sparse(num_item, num_user);
repeat_record = containers.Map();

for i = 1:size(rating,1)
    userID = rating(i,1);
    itemID = rating(i,2);
    value = rating(i,3);
    
    colID = map_from_userID_to_colID(userID);
    rowID = map_from_itemID_to_rowID(itemID);
    [R, repeat_record] = put_value_into_matrix(rowID, colID, value, R, repeat_record);
end

R = use_mean_for_multiple_records(R, repeat_record);

%% Transform the raw trust data to trust matrix S
S = sparse(num_user, num_user);
repeat_record = containers.Map();

for i = 1:size(trust,1)
    trustor = trust(i,1);
    trustee = trust(i,2);
    value = trust(i,3);
    rowID = map_from_userID_to_colID(trustee);
    colID = map_from_userID_to_colID(trustor);
    [S, repeat_record] = put_value_into_matrix(rowID, colID, value, S, repeat_record);
end

S = use_mean_for_multiple_records(S, repeat_record);

%% Save data to mat format

save('data/dataname.mat', 'R', 'S', 'map');

end

%% Utility funtions
function [A, repeat_record] = put_value_into_matrix(rowID, colID, value, A, repeat_record)
    if A(rowID, colID) > 0 
        key = mat2str([rowID,colID]);
        if repeat_record.isKey(key)
            repeat_record(key) = repeat_record(key) + 1;
        else
            repeat_record(key) = 2;
        end
    end
    A(rowID, colID) = A(rowID, colID) + value;
end

function A = use_mean_for_multiple_records(A, repeat_record)
    for key = repeat_record.keys
        key_str = key{1};
        key = str2num(key_str);
        rowID = key(1);
        colID = key(2);
        times = repeat_record(key_str);
        A(rowID, colID) = A(rowID, colID) / times;
    end
end
