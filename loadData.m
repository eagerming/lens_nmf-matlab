function  [R,social_matrix,map,item_matrix]= loadData(rating_filePath, trust_filePath, dataname, item_network_filePath)
%% read filmtrust data:

% rating_filePath = 'data/filmtrust/rating/ratings.txt';
% trust_filePath = 'data/filmtrust/trust/trust.txt';

rating_fileID = fopen(rating_filePath);
trust_fileID = fopen(trust_filePath);


rating = textscan(rating_fileID,'%f %f %f');
trust = textscan(trust_fileID,'%f %f %f');

rating = [rating{1,1}, rating{1,2}, rating{1,3}];
trust = [trust{1,1}, trust{1,2}, trust{1,3}];

if exist('item_network_filePath', 'var')
    item_network_fileID = fopen(item_network_filePath);
    item_network = textscan(item_network_fileID,'%f %f %f');
    item_network = [item_network{1,1}, item_network{1,2}, item_network{1,3}];
end

%% Mapping

userID = trust(:,1:2);
userID = [userID(:); rating(:,1)];
map_from_colID_to_userID = unique(userID,'sorted');
map_from_userID_to_colID = containers.Map(map_from_colID_to_userID, 1:length(map_from_colID_to_userID));

itemID = rating(:,2);
if exist('item_network_filePath', 'var')
    itemID_add = item_network(:,1:2);
    itemID = [itemID; itemID_add(:)];
end
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

%% Transform the raw trust data to trust matrix social_matrix
social_matrix = sparse(num_user, num_user);
repeat_record = containers.Map();

for i = 1:size(trust,1)
    trustor = trust(i,1);
    trustee = trust(i,2);
    value = trust(i,3);
    rowID = map_from_userID_to_colID(trustee);
    colID = map_from_userID_to_colID(trustor);
    [social_matrix, repeat_record] = put_value_into_matrix(rowID, colID, value, social_matrix, repeat_record);
end

social_matrix = use_mean_for_multiple_records(social_matrix, repeat_record);

%% Transform the item network data to item matrix item_matrix
if exist('item_network_filePath', 'var')
    item_matrix = sparse(num_item, num_item);
    repeat_record = containers.Map();

    for i = 1:size(item_network,1)
        trustor_item = item_network(i,1);
        trustee_item = item_network(i,2);
        value = item_network(i,3);
        rowID = map_from_itemID_to_rowID(trustee_item);
        colID = map_from_itemID_to_rowID(trustor_item);
        [item_matrix, repeat_record] = put_value_into_matrix(rowID, colID, value, item_matrix, repeat_record);
    end

    item_matrix = use_mean_for_multiple_records(item_matrix, repeat_record);
end

%% Save data to mat format

if exist('item_network_filePath', 'var')
    save(strcat('data/',dataname), 'R', 'social_matrix', 'map' , 'item_matrix');
else
    save(strcat('data/',dataname), 'R', 'social_matrix', 'map');
end

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
