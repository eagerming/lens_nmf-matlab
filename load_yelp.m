rating_dataPath = 'data/yelp_dataset/yelp_academic_dataset_review.json';
social_network_path = 'data/yelp_dataset/yelp_academic_dataset_user.json';
item_network_path = 'data/yelp_dataset/yelp_academic_dataset_business.json';

%% Prepare to read data


reading_line = 200;
partial = true;

if isinf(reading_line)
    partial = false;
    rating_data = importdata(rating_dataPath);
    social_network_data = importdata(social_network_path);
    item_network_data = importdata(item_network_path);
end

%% Read rating data


map_from_user_to_colID = containers.Map();
if partial
    map_from_colID_to_user = cell(1000000,1);
else
    map_from_colID_to_user = cell(length(social_network_data),1);
end
map_from_item_to_rowID = containers.Map();
if partial
    map_from_rowID_to_item = cell(1000000,1);
    map_from_rowID_to_category = cell(1000000,1);
else
    map_from_rowID_to_item = cell(length(item_network_data),1);
    map_from_rowID_to_category = cell(length(item_network_data),1);
end



line = 0;
colID = 1;
rowID = 1;

if partial
    item_user_list = zeros(10000,3);
else
    item_user_list = zeros(length(rating_data),3);
end

if partial
    fid = fopen(rating_dataPath, 'r');
end


tic;
while true
    if partial
        if feof(fid) || line >= reading_line
            break;
        end
        line_str = fgetl(fid);
        line_data = jsondecode(line_str);
    else
        if line >= length(rating_data)
            break;
        end
        line_str = rating_data{line};
        line_data = line_str;
    end
    
    line = line + 1;
    item = line_data.business_id;
    user = line_data.user_id;
    star = line_data.stars;

    if ~isKey(map_from_user_to_colID, user)
        map_from_colID_to_user{colID} = user;
        map_from_user_to_colID(user) = colID;
        col_this = colID;
        colID = colID + 1;
    else
        col_this = map_from_user_to_colID(user);
    end

    if ~isKey(map_from_item_to_rowID, item)
        map_from_rowID_to_item{rowID} = item;
        map_from_item_to_rowID(item) = rowID;
        row_this = rowID;
        rowID = rowID + 1;
    else
        row_this = map_from_item_to_rowID(item);
    end

    item_user_list(line, :) = [row_this, col_this, star];
end

if partial
    fclose(fid);
end

item_user_list(line+1:end,:) = [];
map_from_colID_to_user(colID:end) = [];
map_from_rowID_to_item(rowID:end) = [];

R = sparse(item_user_list(:,1), item_user_list(:,2), item_user_list(:,3));
mask = sparse(item_user_list(:,1), item_user_list(:,2), ones(size(item_user_list,1),1));

time = toc;
fprintf('Reading rating data lines:[%d], time used[%f]\n', reading_line, time);



%% reading Item network


% item1_item2_list = zeros(length(item_network_data),2);

tic;
if partial
    fid = fopen(item_network_path, 'r');
end

lat_list = zeros(length(map_from_rowID_to_item),1);
lon_list = zeros(length(map_from_rowID_to_item),1);

line = 0;
while true
    if partial
        if feof(fid) || line >= reading_line
            break;
        end
        line_str = fgetl(fid);
        line_data = jsondecode(line_str);
    else
        if line >= length(item_network_data)
            break;
        end
        line_str = item_network_data{line};
        line_data = line_str;
    end
    
    line = line + 1;
    item = line_data.business_id;
    lat = line_data.latitude;
    lon = line_data.longitude;
    category = line_data.categories;
    
    if isKey(map_from_item_to_rowID, item)
        rowID = map_from_item_to_rowID(item);
        map_from_rowID_to_category{rowID} = category;
        lat_list(rowID) = lat;
        lon_list(rowID) = lon;
    end
end

if partial
    fclose(fid);
end
% 
% item1_item2_list(line+1:end,:) = [];
% 
% item_matrix = sparse(item1_item2_list(:,1), item1_item2_list(:,2), ones(size(item1_item2_list,1),1));

lat = lat_list;
lon = lon_list;

sample_portion = 0.01;

num_spot = length(lat);
num_base = round(num_spot * sample_portion);

rng('default')
base_id = datasample(1:num_spot, num_base,'Replace', false);

lat_base = lat(base_id);
lon_base = lon(base_id);

base = [lat_base, lon_base];
lat_lon = [lat, lon];
distance_eculidean = pdist2(base, lat_lon); 
edge_exist = (distance_eculidean <= threshold);

edge = zeros(10000000, 2);
iii = 1;
for i = 1:size(edge_exist,1)
    poi_base = base_id(i);
    num_neighbors = sum(edge_exist(i,:));
    if iii + num_neighbors - 1 > size(edge,1)
        edge = [edge; zeros(size(edge,1),2)];
    end
    edge(iii : iii + num_neighbors - 1,:) = [poi_base * ones(num_neighbors, 1), find(edge_exist(i,:))'];
    iii = iii + num_neighbors;
end
edge(iii:end, :) = [];
item_matrix = sparse(edge(:,1), edge(:,2), ones(size(edge,1),1));
time = toc;
fprintf('Reading rating data lines:[%d], time used[%f]\n', reading_line, time);

%% Reading social data

tic;
if partial
    fid = fopen(social_network_path, 'r');
end


line = 0;
edge = zeros(10000000,2);
iii = 0;
while true
    if partial
        if feof(fid) || line >= reading_line
            break;
        end
        line_str = fgetl(fid);
        line_data = jsondecode(line_str);
    else
        if line >= length(item_network_data)
            break;
        end
        line_str = item_network_data{line};
        line_data = line_str;
    end
    
    line = line + 1;
    user = line_data.user_id;
    userID = map_from_user_to_colID(user);
    friends = line_data.friends;
    
    iii = iii + 1;
    
    if (iii + length(friends) - 1) > size(edge,1)
        edge = [edge; zeros(size(edge,1),2)];
    end
    edge(iii: iii + length(friends) - 1, 1) = userID * ones(length(friends),1);
    iii = iii - 1; 
    if ~strcmp(friends, 'None')
        friends = split(friends, ', ');
        for i = 1:length(friends)
            friend = friends{i};
            friendID = map_from_user_to_colID(friend);
            iii = iii + 1;
            edge(iii, 2) = friendID;
        end
    end
    
    if isKey(map_from_item_to_rowID, item)
        rowID = map_from_item_to_rowID(item);
        map_from_rowID_to_category{rowID} = category;
        lat_list(rowID) = lat;
        lon_list(rowID) = lon;
    end
end
edge(iii+1,:) = [];

social_matrix = sparse(edge(:,1), edge(:,2), ones(size(edge,1),1));

time = toc;
fprintf('Reading rating data lines:[%d], time used[%f]\n', reading_line, time);

%% Data saving

save('data/gowalla.mat', 'social_matrix', 'R', 'mask','item_matrix');
fprintf('Data saved successully!\n');

